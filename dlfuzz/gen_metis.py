import h5py
import tensorflow as tf

from dlfuzz.utils_gen_metis import *
import os
import time

#tf.compat.v1.disable_eager_execution()

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    v = v.reshape(v.shape[0], 28, 28, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v


def run_dlfuzz(model_name, label, model, input_tensor, starting_seeds, imgs_to_sample, run_folder):
    
    # load multiple models sharing the same input tensor
    K.set_learning_phase(0)
    
    model_layer_times1 = init_coverage_times(model)  # times of each neuron covered
    model_layer_times2 = init_coverage_times(model)  # update when new image and adversarial images found
    model_layer_value1 = init_coverage_value(model)
    # start gen inputs

    # e.g.[0,1,2] None for neurons not covered, 0 for covered often, 1 for covered rarely, 2 for high weights
    neuron_select_strategy = ['1']
    threshold = 0.75
    neuron_to_cover_num = 10
    iteration_times = 3

    neuron_to_cover_weight = 0.5
    predict_weight = 0.5
    learning_step = 0.02

    total_time = 0
    total_norm = 0
    adversial_num = 0

    total_perturb_adversial = 0

    i = 0
    SEEDCOUNT = 0
    while SEEDCOUNT < len(starting_seeds) and adversial_num < imgs_to_sample:

        start_time = time.time()

        img_list = []

        orig_label = label
        img_name = 'image_' + str(SEEDCOUNT) + '_label_' + str(orig_label)

        tmp_img = reshape(starting_seeds[SEEDCOUNT])

        count = 0
        SEEDCOUNT += 1

        orig_img = tmp_img.copy()

        img_list.append(tmp_img)

        update_coverage(tmp_img, model, model_layer_times2, threshold)

        while len(img_list) > 0:

            gen_img = img_list[0]

            img_list.remove(gen_img)

            # first check if input already induces differences
            pred1 = model.predict(gen_img)
            label1 = np.argmax(pred1[0])
            if label1 != label:
                continue

            label_top5 = np.argsort(pred1[0])[-5:]

            update_coverage_value(gen_img, model, model_layer_value1)
            update_coverage(gen_img, model, model_layer_times1, threshold)

            orig_label = label1
            orig_pred = pred1

            loss_1 = K.mean(model.get_layer('before_softmax').output[..., orig_label])
            loss_2 = K.mean(model.get_layer('before_softmax').output[..., label_top5[-2]])
            loss_3 = K.mean(model.get_layer('before_softmax').output[..., label_top5[-3]])
            loss_4 = K.mean(model.get_layer('before_softmax').output[..., label_top5[-4]])
            loss_5 = K.mean(model.get_layer('before_softmax').output[..., label_top5[-5]])

            layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)

            # neuron coverage loss
            loss_neuron = neuron_selection(model, model_layer_times1, model_layer_value1, neuron_select_strategy,
                                        neuron_to_cover_num, threshold)
            # loss_neuron = neuron_scale(loss_neuron) # useless, and negative result


            layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

            # for adversarial image generation
            final_loss = K.mean(layer_output)

            # we compute the gradient of the input picture wrt this loss
            #grads = normalize(K.gradients(final_loss, input_tensor)[0])
            grads = normalize(_compute_gradients(final_loss, [input_tensor])[0])
            grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
            grads_tensor_list.extend(loss_neuron)
            grads_tensor_list.append(grads)
            # this function returns the loss and grads given the input picture
            

            iterate = K.function(input_tensor, grads_tensor_list)

            # we run gradient ascent for 3 steps
            for iters in range(iteration_times):

                loss_neuron_list = iterate([gen_img])
                perturb = loss_neuron_list[-1] * learning_step
                import matplotlib.pyplot as plt
                #plt.imsave('orig_img.png', gen_img.reshape(28, 28), cmap='gray', format='png')
                gen_img += perturb

                #plt.imsave('perturb1.png', perturb.reshape(28, 28), cmap='gray', format='png')
                #plt.imsave('pert_img.png', gen_img.reshape(28, 28), cmap='gray', format='png')
                #print(perturb.shape)
                #exit()

                #previous accumulated neuron coverage
                previous_coverage = neuron_covered(model_layer_times1)[2]

                pred1 = model.predict(gen_img)
                label1 = np.argmax(pred1[0])

                update_coverage(gen_img, model, model_layer_times1, threshold) # for seed selection

                current_coverage = neuron_covered(model_layer_times1)[2]

                diff_img = gen_img - orig_img

                L2_norm = np.linalg.norm(diff_img)

                orig_L2_norm = np.linalg.norm(orig_img)

                perturb_adversial = L2_norm / orig_L2_norm

                if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < 0.02:
                    img_list.append(gen_img)

                if label1 != orig_label:
                    update_coverage(gen_img, model, model_layer_times2, threshold)

                    total_norm += L2_norm

                    total_perturb_adversial += perturb_adversial

                    save_img = os.path.join(run_folder, img_name + '_' + str(count))

                    np.save(save_img, gen_img)

                    count += 1

                    adversial_num += 1
        i += 1

        end_time = time.time()

        print('covered neurons percentage %d neurons %.3f'
            % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

        duration = end_time - start_time

        print('used time : ' + str(duration))

        total_time += duration

    print('covered neurons percentage %d neurons %.3f'
        % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

    summary_file = os.path.join(run_folder, "summary.txt")

    with open(summary_file, 'w') as f:
        f.write(f"----GENERATION----\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: MNIST\n")
        f.write(f"Images evaluated: {imgs_to_sample}\n")
        f.write(f"Generation time: {total_time}\n")
        f.write(f"Adversial num: {adversial_num}\n")
        if adversial_num != 0:
            f.write(f"Avarage norm: {total_norm / adversial_num}\n")
            f.write(f"Average perb adversial: {total_perturb_adversial / adversial_num}\n")
        f.write(f"\n")