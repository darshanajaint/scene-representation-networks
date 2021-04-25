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
p.add_argument('--img_sidelengths', type=int, default='64', required=False,
               help='Progression of image sidelengths.'
                    'If comma-separated list, will train on each sidelength for respective max_steps.'
                    'Images are downsampled to the respective resolution.')
p.add_argument('--max_steps_per_img_sidelength', type=int, default="200000",
               help='Maximum number of optimization steps.'
                    'If comma-separated list, is understood as steps per image_sidelength.')
p.add_argument('--batch_size_per_img_sidelength', type=int, default="64",
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

# GAN Options
p.add_argument('--num_instances', type=int, default=1000,
               help='Number of instances that the model was trained with.')
p.add_argument('--batch_num_instances', type=int, default=100,
               help='Number of objects to train on.')
p.add_argument('--num_observations', type=int, default=17,
               help='Number of observations per object to use.')
p.add_argument('--model_type', type=str,
               help='One of "mobilenet", "resnet", and "googlenet".')
p.add_argument('--gan_iterations', type=int, default=100000)
p.add_argument('--gan_start', type=int, default=0)
p.add_argument('--gan_checkpoint', type=str, default=None)


opt = p.parse_args()


def load_model(path, model, model_type):
    state = torch.load(path)
    model.load_state_dict(state[model_type])
    return model


def set_up_generator():
    model = SRNsModel(
        num_instances=opt.num_instances,
        latent_dim=256,
        has_params=False,
        fit_single_srn=False,
        use_unet_renderer=False,
        tracing_steps=10,
        freeze_networks=False,
        discriminator=None,
        freeze_partial=True
    )

    if opt.gan_checkpoint is not None:
        print("Loading from gan checkpoint:", opt.gan_checkpoint)
        model = load_model(opt.gan_checkpoint, model, "generator")
    elif opt.checkpoint_path is not None:
        print("Loading model from %s" % opt.checkpoint_path)
        util.custom_load(model, path=opt.checkpoint_path,
                         discriminator=None,
                         optimizer=None,
                         overwrite_embeddings=False)

    ckpt_dir = os.path.join(opt.logging_root, 'checkpoints')
    models_dir = os.path.join(ckpt_dir, 'models')
    results_dir = os.path.join(ckpt_dir, 'results')

    util.cond_mkdir(opt.logging_root)
    util.cond_mkdir(ckpt_dir)
    util.cond_mkdir(models_dir)
    util.cond_mkdir(results_dir)

    # Save command-line parameters log directory.
    with open(os.path.join(opt.logging_root, "params.txt"),
              "w") as out_file:
        out_file.write('\n'.join(
            ["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(opt.logging_root, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    return model, models_dir, results_dir


def set_up_discriminator(device):
    model = get_model(opt.model_type)
    if opt.gan_checkpoint is not None:
        print("Loading discriminator from gan checkpoint:", opt.gan_checkpoint)
        model = load_model(opt.gan_checkpoint, model, "discriminator")
    transform = get_transform()
    model = ModelUtil(model, transform, device)
    return model


def checkpoint(models_dir, results_dir, iter, disc, disc_results, gen,
               gen_loss):
    models_path = models_dir + '/iter_{:06d}.pth'.format(iter)
    models = {
        'discriminator': disc.model.state_dict(),
        'generator': gen.state_dict()
    }
    torch.save(models, models_path)

    results_path = results_dir + '/iter_{:06d}.pth'.format(iter)
    results = {
        'discriminator': disc_results,
        'generator': gen_loss
    }
    torch.save(results, results_path)


def gan_training(start, num_iterations, discriminator, generator, gen_optimizer,
                 models_dir, results_dir):
    generator.train()
    generator.cuda()
    for iter in range(start, num_iterations):
        # sample images from test set
        # generate fakes for sample images - fakes = gen_model(samples)
        # discrim_preds = discrim_model([samples, fakes, labels])
        # discrim_loss = formula #1
        # discrim_loss.backward, discrim_optimizer.step
        # freeze discriminator
        # new_fake_pred = discrim_model([fakes, labels])
        # gen_loss = formula #2
        # gen_loss.backward, gen_optimizer.step

        samples = dataio.SceneClassDataset.generate_batch(
            data_root=opt.data_root,
            num_observations=opt.num_observations,
            num_instances=opt.batch_num_instances,
            img_sidelength=opt.img_sidelengths,
        )

        samples = DataLoader(
            samples,
            batch_size=opt.batch_size_per_img_sidelength,
            shuffle=True,
            drop_last=True,
            collate_fn=samples.collate_fn,
            pin_memory=opt.preload
        )

        # fakes = []
        # reals = []
        batch_iter = 0
        running_loss = 0
        disc_res = {}
        for generator_input, ground_truth in samples:
            generator_output = generator(generator_input)
            output_imgs = generator.get_output_img(generator_output)
            true_imgs = util.lin2img(ground_truth['rgb'])
            fakes = output_imgs  # .clone()
            # fakes = fakes.detach().cpu() # .numpy() # .astype(np.uint8)
            # += list(output_imgs)  # .detach().cpu().numpy())
            reals = true_imgs  # .clone()
            # reals = reals.detach().cpu() # .numpy()# .astype(np.uint8)
            # += list(true_imgs)  # .detach().cpu().numpy())

            # Discriminator training
            disc_res = discriminator.train(reals, fakes)

            # Generator training
            # fakes = output_imgs  # .to(torch.uint8)
            gen_optimizer.zero_grad()
            generator.set_discriminator(discriminator)
            gen_loss = generator.get_gan_loss(fakes)
            gen_loss.backward()
            gen_optimizer.step()

            running_loss += gen_loss.item()
            batch_iter += 1

        checkpoint(models_dir, results_dir, iter, discriminator, disc_res,
                   generator, running_loss / batch_iter)


def save_predictions(path, pred, idx):
    path = path + '/iter_{:06d}.pth'.format(idx)
    torch.save(pred, path)


def make_predictions(discriminator, generator, results_dir):
    dataset = dataio.SceneClassDataset.generate_dataset(
        root_dir=opt.data_root,
        max_num_instances=opt.max_num_instances_train,
        specific_observation_idcs=opt.specific_observation_idcs,
        max_observations_per_instance=-1,
        samples_per_instance=1,
        img_sidelength=opt.img_sidelengths
    )

    data_loader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )

    with torch.no_grad():
        generator.eval()
        generator.cuda()

        idx = 0
        predictions = list()
        for gen_input, ground_truth in data_loader:
            gen_output = generator(gen_input)

            # get images
            fake = generator.get_output_img(gen_output)
            real = util.lin2img(ground_truth['rgb'])

            # run through discriminator
            result = discriminator.predict_proba(real, fake)
            predictions.append(result)

            # save discriminator results
            idx += 1
            if idx % 500 == 0:
                save_predictions(results_dir, predictions, idx)
                predictions = list()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = set_up_discriminator(device)
    generator, models_dir, results_dir = set_up_generator()
    results_dir = opt.logging_root
    # optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    # gan_training(opt.gan_start, opt.gan_iterations, discriminator, generator,
    #              optimizer, models_dir, results_dir)
    make_predictions(discriminator, generator, results_dir)


if __name__ == "__main__":
    main()
