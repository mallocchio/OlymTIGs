#!/usr/bin/env python
# coding: utf-8

import os
import shutil
from datetime import datetime

import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms
from tqdm import trange

from model import VAE
from Converter import ModelConverter
from Classifier import TorchClassifier, TensorflowClassifier


class GeneticAlgorithm:
    def __init__(self, label, device, vae_model_path, classifier_model_path):
        self.label = label
        self.device = device
        self.vae_model_path = vae_model_path
        self.classifier_model_path = classifier_model_path
        self.img_size = 28 * 28 * 1
        self.h_size = 1600
        self.z_size = 400
        self.gen_num = 500
        self.pop_size = 50
        self.best_left = 20
        self.mut_size = 0.1
        self.imgs_to_samp = 100

    def load_models(self):
        self.vae = VAE(img_size=28 * 28, h_dim=self.h_size, z_dim=self.z_size)
        self.vae.load_state_dict(torch.load(self.vae_model_path, map_location=self.device))
        self.vae.to(self.device)

        if self.classifier_model_path.endswith('.h5'):
            print('Converting tensorflow classifier to torch...\n')
            percorso_originale = self.classifier_model_path
            nuovo_nome_file = "converted_model.pt"
            directory = os.path.dirname(percorso_originale)
            nuovo_percorso = os.path.join(directory, nuovo_nome_file)

            converter = ModelConverter(self.classifier_model_path, nuovo_percorso)
            tensorflow_classifier = converter.load_tensorflow_model()
            torch_classifier = converter.convert_to_pytorch(tensorflow_classifier)
            converter.save_pytorch_model(torch_classifier)
            self.classifier_model_path = nuovo_percorso
            print('Conversion ended successfully...\n')
            self.classifier = TorchClassifier()
            print('Classifier initialized...\n')

        elif self.classifier_model_path.endswith('.pt'):
            self.classifier = TorchClassifier()
            print('Classifier initialized...\n')

        self.classifier.load_model(self.classifier_model_path)
        print('Model loaded...\n')

    def prepare_data_loader(self):
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),
                                                   download=True)

        if self.label != -1:
            idx = test_dataset.targets == self.label
            idx = np.where(idx)[0]
            subset = Subset(test_dataset, idx)
            self.test_data_loader = torch.utils.data.DataLoader(dataset=subset, batch_size=1, shuffle=True)
        else:
            self.test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
        print("Data loader ready...")

    def run_genetic_algorithm(self):
        all_img_lst = []

        for img_idx in trange(self.imgs_to_samp):
            for i, (x, x_class) in enumerate(self.test_data_loader):
                samp_img = x[0:1]
                samp_class = x_class[0].item()

            img_enc, _ = self.vae.encode(samp_img.view(-1, self.img_size).to(self.device))

            init_pop = [img_enc + 0.7 * torch.randn(1, 400).to(self.device) for _ in range(self.pop_size)]
            now_pop = init_pop
            prev_best = 999
            binom_sampler = torch.distributions.binomial.Binomial(probs=0.5 * torch.ones(img_enc.size()))

            for g_idx in range(self.gen_num):
                indivs = torch.cat(now_pop, dim=0)
                dec_imgs = self.vae.decode(indivs).view(-1, 1, 28, 28)
                all_logits = self.classifier.get_batch_prediction(dec_imgs)

                indv_score = [999 if samp_class == torch.argmax(all_logits[i_idx]).item()
                              else torch.sum(torch.abs(indivs[i_idx] - img_enc))
                              for i_idx in range(self.pop_size)]

                best_idxs = sorted(range(len(indv_score)), key=lambda i: indv_score[i], reverse=True)[-self.best_left:]
                now_best = min(indv_score)

                if now_best == prev_best:
                    self.mut_size *= 0.7
                else:
                    self.mut_size = 0.1
                parent_pop = [now_pop[idx] for idx in best_idxs]

                k_pop = []
                for k_idx in range(self.pop_size - self.best_left):
                    mom_idx, pop_idx = np.random.choice(self.best_left, size=2, replace=False)
                    spl_idx = np.random.choice(400, size=1)[0]
                    k_gene = torch.cat([parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]], dim=1)

                    diffs = (k_gene != img_enc).float()
                    k_gene += self.mut_size * torch.randn(k_gene.size()).to(self.device) * diffs
                    interp_mask = binom_sampler.sample().to(self.device)
                    k_gene = interp_mask * img_enc + (1 - interp_mask) * k_gene

                    k_pop.append(k_gene)
                now_pop = parent_pop + k_pop
                prev_best = now_best
                if self.mut_size < 1e-3:
                    break

            mod_best = parent_pop[-1].clone()
            final_bound_img = self.vae.decode(parent_pop[-1])
            final_bound_img = final_bound_img.detach().cpu().numpy()
            all_img_lst.append(final_bound_img)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_folder = f'run_{timestamp}'
        os.makedirs(run_folder)

        all_imgs = np.vstack(all_img_lst)
        np.save(os.path.join(run_folder, f'bound_imgs_MNIST_{self.label}.npy'), all_imgs)
        print("Generated inputs:", len(all_img_lst))
        print("FINISH")

        results_folder = "Bound_images_results"
        os.makedirs(results_folder, exist_ok=True)
        shutil.move(run_folder, os.path.join(results_folder, run_folder))


if __name__ == "__main__":
    label = -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model_path = "vae_model.pt"
    classifier_model_path = "classifier_model.h5"

    ga = GeneticAlgorithm(label, device, vae_model_path, classifier_model_path)
    ga.load_models()
    ga.prepare_data_loader()
    ga.run_genetic_algorithm()
