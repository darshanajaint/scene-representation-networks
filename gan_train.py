import torch.cuda

import dataio
import configargparse
import util
import os

from srns import *
from torch.utils.data import DataLoader
from discriminator.util import *
from discriminator.ModelUtil import ModelUtil

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True,
      help='Path to config file.')

# Multi-resolution training: Instead of passing only a single value, each of these command-line arguments take comma-
# separated lists. If no multi-resolution training is required, simply pass single values (see default values).
p.add_argument('--img_sidelengths', type=str, default='64', required=False,
               help='Progression of image sidelengths.'
                    'If comma-separated list, will train on each sidelength for respective max_steps.'
                    'Images are downsampled to the respective resolution.')
p.add_argument('--max_steps_per_img_sidelength', type=str, default="200000",
               help='Maximum number of optimization steps.'
                    'If comma-separated list, is understood as steps per image_sidelength.')
p.add_argument('--batch_size_per_img_sidelength', type=str, default="64",
               help='Training batch size.'
                    'If comma-separated list, will train each image sidelength with respective batch size.')

# Training options
p.add_argument('--data_root', required=True,
               help='Path to directory with training data.')
p.add_argument('--val_root', required=False,
               help='Path to directory with validation data.')
p.add_argument('--logging_root', type=str, default='./logs',
               required=False,
               help='path to directory where checkpoints & tensorboard events will be saved.')

p.add_argument('--lr', type=float, default=5e-5,
               help='learning rate. default=5e-5')

p.add_argument('--l1_weight', type=float, default=200,
               help='Weight for l1 loss term (lambda_img in paper).')
p.add_argument('--kl_weight', type=float, default=1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')
p.add_argument('--reg_weight', type=float, default=1e-3,
               help='Weight for depth regularization term (lambda_depth in paper).')

p.add_argument('--steps_til_ckpt', type=int, default=10000,
               help='Number of iterations until checkpoint is saved.')
p.add_argument('--steps_til_val', type=int, default=1000,
               help='Number of iterations until validation set is run.')
p.add_argument('--no_validation', action='store_true', default=False,
               help='If no validation set should be used.')

p.add_argument('--preload', action='store_true', default=False,
               help='Whether to preload data to RAM.')

p.add_argument('--checkpoint_path', default=None,
               help='Checkpoint to trained model.')
p.add_argument('--overwrite_embeddings', action='store_true', default=False,
               help='When loading from checkpoint: Whether to discard checkpoint embeddings and initialize at random.')
p.add_argument('--start_step', type=int, default=0,
               help='If continuing from checkpoint, which iteration to start counting at.')

p.add_argument('--specific_observation_idcs', type=str, default=None,
               help='Only pick a subset of specific observations for each instance.')

p.add_argument('--max_num_instances_train', type=int, default=-1,
               help='If \'data_root\' has more instances, only the first max_num_instances_train are used')
p.add_argument('--max_num_observations_train', type=int, default=50,
               required=False,
               help='If an instance has more observations, only the first max_num_observations_train are used')
p.add_argument('--max_num_instances_val', type=int, default=10, required=False,
               help='If \'val_root\' has more instances, only the first max_num_instances_val are used')
p.add_argument('--max_num_observations_val', type=int, default=10,
               required=False,
               help='Maximum numbers of observations per validation instance')

p.add_argument('--has_params', action='store_true', default=False,
               help='Whether each object instance already comes with its own parameter vector.')

# Model options
p.add_argument('--tracing_steps', type=int, default=10,
               help='Number of steps of intersection tester.')
p.add_argument('--freeze_networks', action='store_true',
               help='Whether to freeze weights of all networks in SRN (not the embeddings!).')
p.add_argument('--fit_single_srn', action='store_true', required=False,
               help='Only fit a single SRN for a single scene (not a class of SRNs) --> no hypernetwork')
p.add_argument('--use_unet_renderer', action='store_true',
               help='Whether to use a DeepVoxels-style unet as rendering network or a per-pixel 1x1 convnet')
p.add_argument('--embedding_size', type=int, default=256,
               help='Dimensionality of latent embedding.')

opt = p.parse_args()


def set_up_srns_model(discriminator, freeze_partial):
    img_sidelengths = 64

    train_dataset = dataio.SceneClassDataset(
        root_dir=opt.data_root,
        max_num_instances=-1,
        max_observations_per_instance=50,
        img_sidelength=img_sidelengths,
        specific_observation_idcs=None,
        samples_per_instance=1
    )

    model = SRNsModel(
        num_instances=train_dataset.num_instances,
        latent_dim=256,
        has_params=False,
        fit_single_srn=False,
        use_unet_renderer=False,
        tracing_steps=10,
        freeze_networks=False,
        discriminator=discriminator,
        freeze_partial=freeze_partial
    )
    
    if opt.checkpoint_path is not None:
        print("Loading model from %s" % opt.checkpoint_path)
        util.custom_load(model, path=opt.checkpoint_path,
                         discriminator=None,
                         optimizer=None,
                         overwrite_embeddings=False)

    # TODO: check logging root names for discriminator and generator
    ckpt_dir = os.path.join(opt.logging_root, 'checkpoints')

    util.cond_mkdir(opt.logging_root)
    util.cond_mkdir(ckpt_dir)

    # Save command-line parameters log directory.
    with open(os.path.join(opt.logging_root, "params.txt"),
              "w") as out_file:
        out_file.write('\n'.join(
            ["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(opt.logging_root, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    return model, train_dataset, ckpt_dir


def train_generator(model, train_dataset, ckpt_dir):
    # Parses indices of specific observations from comma-separated list.
    batch_size = opt.batch_size_per_sidelength

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    model.train()
    model.cuda()

    iter = opt.start_step

    print('Beginning training...')
    print("\n" + "#" * 10)
    print("Training generator with batch size %d" % batch_size)
    print("#" * 10 + "\n")

    # Need to instantiate DataLoader every time to set new batch size.
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=train_dataset.collate_fn,
                                  pin_memory=False)

    # Todo: Implement validation set here
    batch_num = 0
    # generated images
    # ground truths
    # optimizer.zero_grad()
    # total_loss = model.get_gan_loss(gen, true)
    # backward
    # step
    for model_input, ground_truth in train_dataloader:
        model_outputs = model(model_input)
        optimizer.zero_grad()
        total_loss = model.get_gan_loss(model_outputs, ground_truth)
        total_loss.backward()
        optimizer.step()

        print("Iter %07d   Batch %03d   L_gan %0.4f" %
              (iter, batch_num, total_loss))
        batch_num += 1

    iter += 1
    util.custom_save(model,
                     os.path.join(ckpt_dir, 'iter_%06d.pth' % iter),
                     discriminator=None,
                     optimizer=optimizer)


def set_up_discriminator(device):
    model = get_model(opt.model_type)
    transform = get_transform()
    model = ModelUtil(model, transform, device, num_epochs=opt.k)
    return model


def train_discriminator(model, train_dataset, ckpt_dir):
    # 21,120 images in train data set
    # batch_size = 128

    model.train()
    return


def checkpoint(path, iter, disc, disc_results, gen, gen_loss):
    path = os.path.join(path, 'iter_%04d.pth' % iter)
    state = {
        'discriminator': disc.state_dict(),
        'discriminator_results': disc_results,
        'generator': gen.state_dict(),
        'generator_loss': gen_loss
    }
    torch.save(state, path)


def gan_training(num_iterations, discriminator, generator, gen_optimizer,
                 ckpt_dir):
    generator.train()
    for iter in range(num_iterations):
        # sample images from test set
        # generate fakes for sample images - fakes = gen_model(samples)
        # discrim_preds = discrim_model([samples, fakes, labels])
        # discrim_loss = formula #1
        # discrim_loss.backward, discrim_optimizer.step
        # freeze discriminator
        # new_fake_pred = discrim_model([fakes, labels])
        # gen_loss = formula #2
        # gen_loss.backward, gen_optimizer.step

        samples = [] # data loader
        fakes = []
        reals = []
        for generator_input, ground_truth in samples:
            generator_output = generator(generator_input)
            fakes.extend(generator_output.cpu().numpy())
            reals.extend(ground_truth.cpu().numpy())

        # Discriminator training
        disc_res = discriminator.train(reals, fakes)

        # Generator training
        gen_optimizer.zero_grad()
        generator.set_discriminator(discriminator)
        gen_loss = generator.get_gan_loss(torch.Tensor(fakes))
        gen_loss.backward()
        gen_optimizer.step()

        checkpoint(ckpt_dir, iter, discriminator, disc_res, generator,
                   gen_loss.item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = set_up_discriminator(device)
    renderer, dataset, ckpt_dir = set_up_srns_model(discriminator, True)
    
