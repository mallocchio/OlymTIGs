#!/usr/bin/env python
# coding: utf-8

# funziona solamente con classificatori torch

import os
import numpy as np
import torch
from tqdm import trange
from torch.distributions.binomial import Binomial
import time

def run_sinvad(model_name, label, device, vae, classifier, test_data_loader, img_rows, img_cols, imgs_to_sample, run_folder):

    ### GA Params ###
    img_size = img_rows * img_cols * 1
    gen_num = 500
    pop_size = 50
    best_left = 20
    mut_size = 0.1
    imgs_to_samp = imgs_to_sample

    count = 1

    with torch.no_grad(): # since nothing is trained here

        start_time = time.time()

        ### multi-image sample loop ###
        for img_idx in trange(imgs_to_samp):
            ### Sample image ###
            for i, (x, x_class) in enumerate(test_data_loader):
                samp_img = x[0:1]
                samp_class = x_class[0].item()

            img_enc, _ = vae.encode(samp_img.view(-1, img_size).to(device))

            ### Initialize optimization ###
            init_pop = [img_enc + 0.7 * torch.randn(1, 400).to(device) for _ in range(pop_size)]
            now_pop = init_pop
            prev_best = 999
            binom_sampler = torch.distributions.binomial.Binomial(probs=0.5*torch.ones(img_enc.size()))

            ### GA ###
            for g_idx in range(gen_num):
                indivs = torch.cat(now_pop, dim=0)
                dec_imgs = vae.decode(indivs).view(-1, 1, img_rows, img_cols)
                all_logits = classifier(dec_imgs)

                #test = [torch.argmax(all_logits[i]) for i in range(pop_size)]

                indv_score = [999 if samp_class == torch.argmax(all_logits[i_idx]).item()
                # indv_score = [999 if all_logits[(i_idx, samp_class)] == max(all_logits[i_idx])
                            else torch.sum(torch.abs(indivs[i_idx] - img_enc))
                            for i_idx in range(pop_size)]

                best_idxs = sorted(range(len(indv_score)), key=lambda i: indv_score[i], reverse=True)[-best_left:]
                now_best = min(indv_score)

                if now_best == prev_best:
                    mut_size *= 0.7
                else:
                    mut_size = 0.1
                parent_pop = [now_pop[idx] for idx in best_idxs]

                k_pop = []
                for k_idx in range(pop_size-best_left):
                    mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                    spl_idx = np.random.choice(400, size=1)[0]
                    k_gene = torch.cat([parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]], dim=1) # crossover

                    # mutation
                    diffs = (k_gene != img_enc).float()
                    k_gene += mut_size * torch.randn(k_gene.size()).to(device) * diffs # random adding noise only to diff places
                    # random matching to img_enc
                    interp_mask = binom_sampler.sample().to(device)
                    k_gene = interp_mask * img_enc + (1 - interp_mask) * k_gene

                    k_pop.append(k_gene)
                now_pop = parent_pop + k_pop
                prev_best = now_best
                if mut_size < 1e-3:
                    break # that's enough and optim is slower than I expected

            mod_best = parent_pop[-1].clone()
            final_bound_img = vae.decode(parent_pop[-1])
            final_bound_img = final_bound_img.detach().cpu().numpy()

            save_img = os.path.join(run_folder, f"image_{count}_label_{label}.npy")
            final_bound_img.reshape((1, img_rows, img_cols, 1))
            np.save(save_img, final_bound_img)

            count = count + 1
            

        #all_imgs = np.vstack(all_img_lst)all

        end_time = time.time()

        #np.save(os.path.join(run_folder, f'bound_imgs_MNIST_{label}.npy'), all_imgs)

        summary_file = os.path.join(run_folder, "summary.txt")

        with open(summary_file, 'w') as f:
            f.write(f"----GENERATION----\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: MNIST\n")
            f.write(f"Images generated: {imgs_to_sample}\n")
            f.write(f"Generation time: {end_time - start_time}\n")
            f.write(f"\n")

