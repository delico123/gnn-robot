import os
import datetime
import logging

import torch
import torch.nn as nn
import wandb

from core.model import build_rstruc_model, build_full_model
from core.optimizer import get_optimizer


# TODO: batch size error (dimension)
# TODO: merge or generalize train/load (capsulize net)

class Solver(nn.Module):
    # TODO: checkpoints
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('cpu')
        
        # W&B Sweep for rs
        if args.rs_sweep:
            config_defaults = {
                'optimizer': args.opt,
                'learning_rate': args.rs_lr,
                'latent_size': args.rs_latent,
                'opt_epsilon': args.opt_eps
            }
            # Init each wandb run
            wandb.init(config=config_defaults)

            # Define config
            config = wandb.config
            
            self.net_rs, config = build_rstruc_model(args, config)
        
        else:
            feat = 2 if args.subtask == 'forward' else 1  # (old forward)

            if args.mode == 'rstruc':
                self.net_rs, config = build_rstruc_model(args)
                self.nets = [self.net_rs]
                self.conv = args.rs_conv

            elif args.mode == 'rmotion':
                self.net_rm, config = build_rstruc_model(args, feat=feat)
                self.nets = [self.net_rm]
                self.conv = args.rs_conv # TODO

            elif args.mode == 'train':
                self.net_rs, config_rs = build_rstruc_model(args)
                self.net_rm, config_rm = build_rstruc_model(args, feat=feat) # TODO
                self.net_full, config = build_full_model(args) # partial full (only mlp)
                self.nets = [self.net_full, self.net_rs, self.net_rm]
                self.conv = args.rs_conv

            else:
                logging.warning(f"{args.mode}")
                raise NotImplementedError

            if args.wnb:
                # Init wandb run
                # name, project, run
                wandb.init(config=config,
                            tags=[self.conv, args.mode],
                            name=self.conv # Run name
                            # project= # Project name. Default: gnn-robot
                            )

                wandb.config.update({'data_simple':args.data_simple,  # True: joint3 only
                                        'wnb_note':args.wnb_note})

                # wandb magic
                wandb.watch(self.nets, log_freq=100)
            
        # Optimizer
        self.optimizer = get_optimizer(args, config, self.nets)

        self.to(self.device)

    def train_reconstruc(self, data_loader):
        """ train structure reconstruction model (rs)
        """
        args = self.args
        net = self.nets[0]
        optimizer = self.optimizer

        train_loader, val_loader = data_loader

        log_latent = []

        for epoch in range(args.rs_epoch+1):
            net.train()

            total_loss = 0
            total_loss_cls = 0
            total_loss_rgr = 0
            log_latent_epoch = []
            for data_set in train_loader:
                if args.mode == "rstruc":
                    data = data_set
                elif args.mode == "rmotion":
                    data = data_set[1]
                    data.x = data.s # state only
                else:
                    data = data_set[1]

                optimizer.zero_grad()

                # TEMP: normalize x[1]
                # if args.rs_dnorm:
                #     data.x[:,1] = data.x[:,1]/0.4 # MIN/MAX Norm
                    # print(data.x)
                    # raise NotImplementedError
                # if args.rs_dnorm:
                #     data.x = data.x[:,0]
                # if args.rs_dnorm:
                #     data.x = data.x[:,1]

                num_node = data.num_nodes

                z = net.encode(data.x, num_node, data.edge_index)
                # output = net.decode(z, num_node, data.edge_index)
                output = net.decode(z, args.node_padding, num_node, data.edge_index) # for debug
                if args.log_save:
                    if epoch % 10 == 0:
                        if args.mode == "rstruc":
                            log_latent_epoch.append([num_node, data.x, output, z])
                        elif args.mode == "rmotion":
                            log_latent_epoch.append([num_node, data_set[0].x, data.x, output, z])

                if args.mode == 'rstruc':
                    loss_cls = net.recon_loss_cls(predict=output[:1][:num_node], target=data.x[:1][:num_node]) # fixed pos
                    loss_rgr = net.recon_loss_rgr(predict=output[1:][:num_node], target=data.x[1:][:num_node])
                    
                    total_loss_cls += loss_cls.item()
                    total_loss_rgr += loss_rgr.item()

                if args.rs_loss_single is False and args.mode == 'rstruc': # Case: loss separated (multitask learning)
                    loss = loss_cls + loss_rgr
                else:
                    loss = net.recon_loss(predict=output[:num_node], target=data.x[:num_node]) # Case: loss single, node only
                    # loss = net.recon_loss(predict=output, target=data.x) # Case: loss single, node all

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            # Eval
            # if eval_per
            with torch.no_grad():
                net.eval()

                eval_loss = 0
                eval_loss_cls = 0
                eval_loss_rgr = 0
                eval_log_latent_epoch = []
                for data_set in val_loader:
                    if args.mode == "rstruc":
                        data = data_set
                    elif args.mode == "rmotion":
                        data = data_set[1]
                        # if args.subtask == "inverse":
                        data.x = data.s # 8
                    else:
                        data = data_set[1]

                    num_node = data.y if self.args.mode == "rstruc" else data_set[0].y

                    z = net.encode(data.x, num_node, data.edge_index)
                    output = net.decode(z, args.node_padding, num_node, data.edge_index)
                    if args.log_save:
                        if epoch % 10 == 0:
                            if args.mode == "rstruc":
                                eval_log_latent_epoch.append([num_node, data.x, output, z])
                            elif args.mode == "rmotion":
                                eval_log_latent_epoch.append([num_node, data_set[0].x, data.x, output, z])
                    
                    if args.mode == 'rstruc':
                        loss_cls = net.recon_loss_cls(predict=output[:1][:num_node], target=data.x[:1][:num_node]) # fixed pos
                        loss_rgr = net.recon_loss_rgr(predict=output[1:][:num_node], target=data.x[1:][:num_node])
                        
                        eval_loss_cls += loss_cls.item()
                        eval_loss_rgr += loss_rgr.item()

                    if args.rs_loss_single is False and args.mode == 'rstruc': # Case: loss separated (multitask learning)
                        loss = loss_cls + loss_rgr
                    else:
                        loss = net.recon_loss(predict=output[:num_node], target=data.x[:num_node]) # Case: loss single, node only
                        # loss = net.recon_loss(predict=output, target=data.x) # Case: loss single, node all

                    eval_loss += loss.item()

            if args.rs_sweep or args.wnb:
                wandb.log({"loss": total_loss/len(train_loader),
                            "loss_cls": total_loss_cls/len(train_loader),
                            "loss_rgr": total_loss_rgr/len(train_loader),
                            "loss-val": eval_loss/len(val_loader),
                            "loss_cls-val": eval_loss_cls/len(val_loader),
                            "loss_rgr-val": eval_loss_rgr/len(val_loader)
                })

            if epoch % args.log_per == 0:
                logging.info(f"Epoch: {epoch}, Loss: {total_loss/len(train_loader)}")
                logging.info(f"Epoch: {epoch}, Loss_val: {eval_loss/len(val_loader)}")
                logging.info(f"Z: {z}, O: {output}, T:{data.x}")

            if epoch % 10 == 0:
                log_latent.append({
                    'epoch': epoch,
                    'train_latent': log_latent_epoch,
                    'val_latent': eval_log_latent_epoch,

                })

        # save model # TODO: ckpt
        if args.wnb:
            wandb.save(os.path.join(wandb.run.dir, 'model.h5'))
        dpath = args.rs_ckpt if args.mode == 'rstruc' else args.rm_ckpt
        now = datetime.datetime.now()
        ppath = os.path.join(dpath, f"pretrain-{args.mode}-{args.rs_conv}-{now.hour}{now.minute}.pth")
        logging.info(f"Saving recon pretrained model.. {ppath}")
        torch.save(net.state_dict(), ppath)

        if args.log_save:
            import pickle
            pickle.dump(log_latent, open(f"./log/latent/pretrain-{args.mode}-{args.rs_conv}-{now.hour}{now.minute}.pkl", "wb"))


    def train(self, data_loader):
        """ train structure reconstruction model (rs)
        """
        args = self.args
        net = self.nets[0] # self.net_full

        if args.rs_conv == 'test_simple_decoder':
            # path_rs = os.path.join(args.rs_ckpt, "pretrain-rstruc-1623.pth") # gt
            path_rs = os.path.join(args.rs_ckpt, "pretrain-rstruc-gt-ls_8.pth") # gt
            # if args.subtask == 'forward':
            #     path_rm = os.path.join(args.rm_ckpt, "pretrain-rmotion-205.pth") # gt forward
            # elif args.subtask == 'inverse':
            #     # path_rm = os.path.join(args.rm_ckpt, "pretrain-rmotion-2116.pth") # gt inverse old
            path_rm = os.path.join(args.rm_ckpt, "pretrain-rmotion-test_simple_decoder-218.pth") # gt inverse
            # else:
                # NotImplementedError

        elif args.rs_conv=='tree':
            path_rs = os.path.join(args.rs_ckpt, "pretrain-rstruc-ours-ls_8.pth") # ours
        #     if args.subtask == 'forward':
        #         path_rm = os.path.join(args.rm_ckpt, "pretrain-rmotion-2030.pth") # ours forward (state + cmd)
        #     elif args.subtask == 'inverse':
            # path_rm = os.path.join(args.rm_ckpt, "pretrain-rmotion-ours-inverse-ls_8.pth") # ours inverse (state)
        #     else:
        #         NotImplementedError
            path_rm = os.path.join(args.rm_ckpt, "pretrain-rmotion-tree-239.pth")

        
        self.net_rs.load_state_dict(torch.load(path_rs))
        self.net_rm.load_state_dict(torch.load(path_rm))

        net_rs = self.net_rs
        net_rm = self.net_rm
        optimizer = self.optimizer

        train_loader, val_loader = data_loader
        logging.debug(f"train: {len(train_loader)}, val: {len(val_loader)}")

        log_latent = []

        for epoch in range(args.rs_epoch+1):
            # Train
            net.train()

            total_loss = 0
            total_loss_f = 0
            total_loss_i = 0
            recon_loss_m = 0
            log_latent_epoch = []
            for data_set in train_loader:

                d_struc = data_set[0]
                d_motion = data_set[1]

                optimizer.zero_grad()

                num_node = d_struc.y

                if args.subtask == 'forward': # note: old forward
                    # (forward)
                    z_struc = net_rs.encode(d_struc.x, num_node, d_struc.edge_index)
                    z_motion = net_rm.encode(d_motion.x, num_node, d_motion.edge_index) # x = s + c (NA in new data)
                    output = net(z_struc, z_motion) # dp: 2
                    
                    loss = net.loss(predict=output[:num_node], target=d_motion.y) # Case: loss single, node only
                
                elif args.subtask == 'inverse':
                    z_struc = net_rs.encode(d_struc.x, num_node, d_struc.edge_index)
                    z_motion = net_rm.encode(d_motion.s, num_node, d_motion.edge_index)
                    ee_pos = d_motion.p # ee pos: 2
                    output = net(z_struc, z_motion, ee_pos) # state: 8
                    
                    loss = net.loss(predict=output[:num_node], target=d_motion.s[:num_node]) # Case: loss single, node only
                
                elif args.subtask == 'multi': 
                    z_struc = net_rs.encode(d_struc.x, num_node, d_struc.edge_index)
                    z_motion = net_rm.encode(d_motion.s, num_node, d_motion.edge_index)
                    
                    out_f, out_i = net(z_struc, z_motion, d_motion.p) # p: ee pos
                    if args.log_save:
                        if epoch % 10 == 0:
                            log_latent_epoch.append([num_node, d_struc.x, d_motion.s, d_motion.p, d_motion.s, out_f, out_i, z_struc, z_motion])
                    
                    loss, loss_forward, loss_inverse = net.loss_multi(predict_f=out_f[:num_node], 
                                            predict_i=out_i[:num_node],
                                            target_f=d_motion.p,
                                            target_i=d_motion.s[:num_node])

                    # if args.loss_decoder:
                    #     # output_s = net_rs.decode(z_struc, args.node_padding, num_node, d_struc.edge_index)
                    #     # loss_dec_structure = 
                        
                    #     output_m = net_rm.decode(z_motion, args.node_padding, num_node, d_motion.edge_index)
                    #     loss_dec_motion = net_rm.recon_loss(output_m[:num_node], d_motion.s[:num_node])
                    #     recon_loss_m += loss_dec_motion.item()

                    #     # loss += 0.3 * (loss_dec_structure + loss_dec_motion)
                    #     loss += 0.3 * loss_dec_motion

                else:
                    raise NotImplementedError

                total_loss += loss.item()
                total_loss_f += loss_forward.item()
                total_loss_i += loss_inverse.item()


                if epoch > 0:
                    loss.backward()
                    optimizer.step()

            # Eval
            with torch.no_grad():
                net.eval()

                eval_loss = 0
                eval_loss_f = 0
                eval_loss_i = 0
                eval_log_latent_epoch = []
                for data_set in val_loader:
                    d_struc = data_set[0]
                    d_motion = data_set[1]

                    num_node = d_struc.y

                    if args.subtask == 'forward':
                        # (forward)
                        z_struc = net_rs.encode(d_struc.x, num_node, d_struc.edge_index)
                        z_motion = net_rm.encode(d_motion.x, num_node, d_motion.edge_index) # x = s + c (NA in new data)
                        output = net(z_struc, z_motion) # dp: 2
                        
                        loss = net.loss(predict=output[:num_node], target=d_motion.y) # Case: loss single, node only
                    
                    elif args.subtask == 'inverse':
                        z_struc = net_rs.encode(d_struc.x, num_node, d_struc.edge_index)
                        z_motion = net_rm.encode(d_motion.s, num_node, d_motion.edge_index)
                        ee_pos = d_motion.p # ee pos: 2
                        output = net(z_struc, z_motion, ee_pos) # state: 8
                        
                        loss = net.loss(predict=output[:num_node], target=d_motion.s[:num_node]) # Case: loss single, node only
                    
                    elif args.subtask == 'multi': 
                        z_struc = net_rs.encode(d_struc.x, num_node, d_struc.edge_index)
                        z_motion = net_rm.encode(d_motion.s, num_node, d_motion.edge_index)
                        
                        out_f, out_i = net(z_struc, z_motion, d_motion.p) # p: ee pos
                        if args.log_save:
                            if epoch % 10 == 0:
                                eval_log_latent_epoch.append([num_node, d_struc.x, d_motion.s, d_motion.p, d_motion.s, out_f, out_i, z_struc, z_motion])
                        
                        loss, loss_forward, loss_inverse = net.loss_multi(predict_f=out_f[:num_node], 
                                                predict_i=out_i[:num_node],
                                                target_f=d_motion.p,
                                                target_i=d_motion.s[:num_node])
                    
                    else:
                        raise NotImplementedError

                    eval_loss += loss.item()
                    eval_loss_f += loss_forward.item()
                    eval_loss_i += loss_inverse.item()

            if args.rs_sweep or args.wnb:
                wandb.log({"loss": total_loss/len(train_loader),
                            "loss_f": total_loss_f/len(train_loader),
                            "loss_i": total_loss_i/len(train_loader),
                            "loss_val": eval_loss/len(val_loader),
                            "loss_val_f": eval_loss_f/len(val_loader),
                            "loss_val_i": eval_loss_i/len(val_loader),
                })

            if epoch % args.log_per == 0:
                logging.info(f"Epoch: {epoch}, Loss: {total_loss/len(train_loader)}")
                logging.info(f"Epoch: {epoch}, Loss_val: {eval_loss/len(val_loader)}")
                logging.info(f"\nO: {out_f}|{out_i}\nT:{d_motion.p}|{d_motion.s}")

            if epoch % 10 == 0:
                log_latent.append({
                    'epoch': epoch,
                    'train_latent': log_latent_epoch,
                    'val_latent': eval_log_latent_epoch,

                })

        # save model # TODO: ckpt
        if args.wnb:
            wandb.save(os.path.join(wandb.run.dir, 'model.h5'))
        # dpath = "./log/train"
        now = datetime.datetime.now()
        # ppath = os.path.join(dpath, f"pretrain-{args.mode}-{args.rs_conv}-{now.hour}{now.minute}.pth")
        # logging.info(f"Saving model.. {ppath}")
        # torch.save(net.state_dict(), ppath)

        if args.log_save:
            import pickle
            pickle.dump(log_latent, open(f"./log/latent/train-{args.mode}-{args.rs_conv}-{now.hour}{now.minute}.pkl", "wb"))