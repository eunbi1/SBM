import os
import logging
import time
import glob
from tkinter import E
import time 
import numpy as np
import tqdm
import torch
import math 
import torch.utils.data as data

import random
import copy


from models.ddpm import Model
from models.ncsnpp import NCSNpp
from models.ema import EMAHelper

from functions.loss import loss_fn
from functions.sde import VPSDE
from functions.sampler import sampler

from datasets import get_dataset

from evaluate.likelihood import get_likelihood_fn
from evaluate.fid_score import fid_score, prdc


from transformers import get_scheduler
import torchvision
import torch.nn.functional as F
import torchvision.utils as tvu
 
class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.sde = VPSDE(config.diffusion.alpha, config.diffusion.beta_schedule)
 

    def train(self):
        args, config = self.args, self.config 
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)

        if args.ddp:
            # Multi-GPU: Revised DataLoader (for Multi-GPU training)
            train_loader = data.DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                sampler=DistributedSampler(dataset)
            ) 
            
            val_loader = data.DataLoader(
                test_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                sampler=DistributedSampler(test_dataset)
            )

        else:
            train_loader = data.DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
            )
            val_loader = data.DataLoader(
                test_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
            )

        if config.model.model_type == "ncsnpp":
            model = NCSNpp(config)
        else:
            model = Model(config)

        if args.ddp:            
            model = model.to(self.device)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        else:
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
        
        
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.lr)
        # optimizer.param_groups[0]["capturable"] = True
        lr_scheduler = get_scheduler(
            "linear",
            # "cosine",
            # "cosine_with_restarts",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=500000,
        )

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"), map_location=self.config.device)
            model.load_state_dict(states[0], strict=False)
            
            try:
                optimizer.load_state_dict(states[1], map_location=self.config.device)
            except: 
                pass
            start_epoch = states[2]
            step = states[3]
            try: 
                lr_scheduler.load_state_dict(states[4])
            except:
                pass 
            if self.config.model.ema:
                ema_helper.load_state_dict(states[-1])
        
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                x = 2 * x - 1
                
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                
                if config.model.is_conditional:
                    y = y.to(self.device)
                else:
                    y = None

                # Noise
      
                e = torch.randn_like(x).to(self.device)
               
                # time
                eps = 1e-5  
                t = torch.rand(n).to(self.device) * (self.sde.T - eps) + eps

                loss = loss_fn(model, self.sde, x, t, e, y)
                
                    
                tb_logger.add_scalar("train_loss", loss, global_step=step)
                logging.info(
                    f"step: {step} with y({y}), loss: {loss.item()}, data time: {data_time / (i+1)}"
                )                
                
                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                
                optimizer.step()
                try:
                    lr_scheduler.step()
                except:
                    pass

                if self.config.model.ema:
                    try:
                        ema_helper.update(model)
                    except:
                        pass

                if step % self.config.training.validation_freq == 0 and args.local_rank == 0:
                    val_loss = 0
                    val_cnt = 0
                    with torch.no_grad():
                        for _, (x, y) in enumerate(val_loader):
                            x = 2 * x - 1
                            
                            n = x.size(0)
                            data_time += time.time() - data_start
                            model.eval()
                            
                            x = x.to(self.device)
                            if config.model.is_conditional:
                                y = y.to(self.device)
                            else:
                                y = None

                            # Noise
                 
                            e = torch.randn_like(x).to(self.device)
          

                            # time           
                            t = torch.rand(n).to(self.device) * (self.sde.T - eps) + eps
                            
                            _loss = loss_fn(model, self.sde, x, t, e, y)
                            val_loss += _loss.item() * n
                            val_cnt += n

                    val_avg_loss = val_loss / val_cnt
                    tb_logger.add_scalar("val_loss", val_avg_loss, global_step=step)
                    logging.info(
                        f"step: {step}, val_loss: {val_avg_loss}"
                    )

                if step % self.config.training.ckpt_store == 0 and args.local_rank == 0:

                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                        lr_scheduler.state_dict()
                    ]
                    
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    self.args.ckpt = os.path.join(self.args.log_path, "ckpt.pth")

                    if args.train_sample and args.local_rank == 0:
                        sample_path = os.path.join(args.exp, "samples")
                        os.makedirs(sample_path, exist_ok=True)
                        model.eval()
                        with torch.no_grad():
                            n = 100
                            x_shape = (n, config.data.channels, config.data.image_size, config.data.image_size)
            
                            t = torch.ones(n, device=self.device) * self.sde.T
                            y = None

                
                            x = torch.randn(x_shape).to(self.device)
                           

                            x = sampler(args, config, x, y, model, self.sde, self.levy,)
                            x = (x + 1) / 2
                            x = x.clamp(0.0, 1.0)
                            
                            name = f"ckpt_{step}_sample.png"
                            tvu.save_image(x, os.path.join(sample_path, name) , nrow = int(np.sqrt(n)))

                data_start = time.time()

    def sample(self):
        args, config = self.args, self.config

        if self.config.model.model_type == "ncsnpp":
            model = NCSNpp(self.config)
        else:
            model = Model(self.config)
        
        if args.ddp:
            model = model.to(self.device)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank]) # find_unused_parameters for fno experiment
        else:
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            
        if "ckpt_dir" in self.config.model.__dict__.keys():
            ckpt_dir = self.config.model.ckpt_dir

            states = torch.load(
                ckpt_dir,
                map_location=self.config.device,
            )

            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                try:
                    ema_helper.ema(model)
                except:
                    pass
            else:
                ema_helper = None
        else:
            raise Exception("Fail to load model due to invalid ckpt_dir")

        model.eval()
        
        # sample generation
    
        print("start generating samples")
        args, config = self.args, self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")

        if args.fid == "train":
            total_n_samples = 50000
        elif args.fid == "test":
            total_n_samples = 10000
        else:
            total_n_samples = config.sampling.batch_size
        
        n_rounds = (total_n_samples) // config.sampling.batch_size
        
        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples"
            ):
                n = config.sampling.batch_size
                
                x_shape = (n, config.data.channels, config.data.image_size, config.data.image_size)
    
               
                x = torch.randn(x_shape).to(self.device)
                
                if config.model.is_conditional:
                    if config.sampling.cond_class is not None:
                        y = torch.ones(n) * config.sampling.cond_class
                        y = y.to(self.device)
                        y = torch.tensor(y, dtype=torch.int64)
                    else:
                        y = None
                else:
                    y = None
                    
                x = LIM_sampler(args, config, x, y, model, self.sde)
                x = x.clamp(-1.0, 1.0)
                x = (x + 1) / 2

                if args.fid == None:
                    # For sampling
                    name = str(time.strftime('%m%d_%H%M_', time.localtime(time.time())))+'.png'
                    tvu.save_image(x, os.path.join(self.args.image_folder, name), nrow= int(np.sqrt(n)))
                    
                else:
                    for i in range(n):
                        tvu.save_image(
                            x[i], os.path.join(args.image_folder, f"{img_id}.png")
                        )
                        img_id += 1

            if args.fid == None:
                pass
                
            elif args.fid == "train":
                dataname = config.data.dataset
                dataset_path = "data/" + dataname.lower() + "_train_fid"
                fid_value = fid_score(dataset_path, args.image_folder, config.sampling.fid_batch_size, self.device, num_workers=0)
                print(f"FID with train dataset : {fid_value}")
                
                prdc(dataset_path, args.image_folder, config.sampling.fid_batch_size, self.device, num_workers=0)
                
            elif args.fid == "test":
                dataname = config.data.dataset
                dataset_path = "data/" + dataname.lower() + "_test_fid"
                fid_value = fid_score(dataset_path, args.image_folder, config.sampling.fid_batch_size, self.device, num_workers=0)
                print(f"FID with test dataset : {fid_value}")
                
                prdc(dataset_path, args.image_folder, config.sampling.fid_batch_size, self.device, num_workers=0)
                
            else:
                pass
