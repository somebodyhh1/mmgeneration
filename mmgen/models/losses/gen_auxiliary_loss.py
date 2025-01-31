# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.autograd as autograd
import torch.distributed as dist
import torch.nn as nn
import torchvision.models.vgg as vgg
from mmcv.runner import load_checkpoint

from mmgen.models.builder import MODULES, build_module
from mmgen.utils import get_root_logger
from .pixelwise_loss import l1_loss, mse_loss


def gen_path_regularizer(generator,
                         num_batches,
                         mean_path_length,
                         pl_batch_shrink=1,
                         decay=0.01,
                         weight=1.,
                         pl_batch_size=None,
                         sync_mean_buffer=False,
                         loss_scaler=None,
                         use_apex_amp=False):
    """Generator Path Regularization.

    Path regularization is proposed in StyelGAN2, which can help the improve
    the continuity of the latent space. More details can be found in:
    Analyzing and Improving the Image Quality of StyleGAN, CVPR2020.

    Args:
        generator (nn.Module): The generator module. Note that this loss
            requires that the generator contains ``return_latents`` interface,
            with which we can get the latent code of the current sample.
        num_batches (int): The number of samples used in calculating this loss.
        mean_path_length (Tensor): The mean path length, calculated by moving
            average.
        pl_batch_shrink (int, optional): The factor of shrinking the batch size
            for saving GPU memory. Defaults to 1.
        decay (float, optional): Decay for moving average of mean path length.
            Defaults to 0.01.
        weight (float, optional): Weight of this loss item. Defaults to ``1.``.
        pl_batch_size (int | None, optional): The batch size in calculating
            generator path. Once this argument is set, the ``num_batches`` will
            be overridden with this argument and won't be affectted by
            ``pl_batch_shrink``. Defaults to None.
        sync_mean_buffer (bool, optional): Whether to sync mean path length
            across all of GPUs. Defaults to False.

    Returns:
        tuple[Tensor]: The penalty loss, detached mean path tensor, and \
            current path length.
    """
    # reduce batch size for conserving GPU memory
    if pl_batch_shrink > 1:
        num_batches = max(1, num_batches // pl_batch_shrink)

    # reset the batch size if pl_batch_size is not None
    if pl_batch_size is not None:
        num_batches = pl_batch_size

    # get output from different generators
    output_dict = generator(None, num_batches=num_batches, return_latents=True)
    fake_img, latents = output_dict['fake_img'], output_dict['latent']

    noise = torch.randn_like(fake_img) / np.sqrt(
        fake_img.shape[2] * fake_img.shape[3])

    if loss_scaler:
        loss = loss_scaler.scale((fake_img * noise).sum())[0]
        grad = autograd.grad(
            outputs=loss,
            inputs=latents,
            grad_outputs=torch.ones(()).to(loss),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        # unsacle the grad
        inv_scale = 1. / loss_scaler.get_scale()
        grad = grad * inv_scale
    elif use_apex_amp:
        from apex.amp._amp_state import _amp_state

        # by default, we use loss_scalers[0] for discriminator and
        # loss_scalers[1] for generator
        _loss_scaler = _amp_state.loss_scalers[1]
        loss = _loss_scaler.loss_scale() * ((fake_img * noise).sum()).float()

        grad = autograd.grad(
            outputs=loss,
            inputs=latents,
            grad_outputs=torch.ones(()).to(loss),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        # unsacle the grad
        inv_scale = 1. / _loss_scaler.loss_scale()
        grad = grad * inv_scale
    else:
        grad = autograd.grad(
            outputs=(fake_img * noise).sum(),
            inputs=latents,
            grad_outputs=torch.ones(()).to(fake_img),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True)[0]
    #刘敬宇修改
    if(grad==None):
        return None,None,None
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    # update mean path
    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    if sync_mean_buffer and dist.is_initialized():
        dist.all_reduce(path_mean)
        path_mean = path_mean / float(dist.get_world_size())

    path_penalty = (path_lengths - path_mean).pow(2).mean() * weight

    return path_penalty, path_mean.detach(), path_lengths


@MODULES.register_module()
class GeneratorPathRegularizer(nn.Module):
    """Generator Path Regularizer.

    Path regularization is proposed in StyelGAN2, which can help the improve
    the continuity of the latent space. More details can be found in:
    Analyzing and Improving the Image Quality of StyleGAN, CVPR2020.

    Users can achieve lazy regularization by setting ``interval`` arguments
    here.

    **Note for the design of ``data_info``:**
    In ``MMGeneration``, almost all of loss modules contain the argument
    ``data_info``, which can be used for constructing the link between the
    input items (needed in loss calculation) and the data from the generative
    model. For example, in the training of GAN model, we will collect all of
    important data/modules into a dictionary:

    .. code-block:: python
        :caption: Code from StaticUnconditionalGAN, train_step
        :linenos:

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g,
            iteration=curr_iter,
            batch_size=batch_size)

    But in this loss, we will need to provide ``generator`` and ``num_batches``
    as input. Thus an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            generator='gen',
            num_batches='batch_size')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        pl_batch_shrink (int, optional): The factor of shrinking the batch size
            for saving GPU memory. Defaults to 1.
        decay (float, optional): Decay for moving average of mean path length.
            Defaults to 0.01.
        pl_batch_size (int | None, optional): The batch size in calculating
            generator path. Once this argument is set, the ``num_batches`` will
            be overridden with this argument and won't be affectted by
            ``pl_batch_shrink``. Defaults to None.
        sync_mean_buffer (bool, optional): Whether to sync mean path length
            across all of GPUs. Defaults to False.
        interval (int, optional): The interval of calculating this loss. This
            argument is used to support lazy regularization. Defaults to 1.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_path_regular'.
    """

    def __init__(self,
                 loss_weight=1.,
                 pl_batch_shrink=1,
                 decay=0.01,
                 pl_batch_size=None,
                 sync_mean_buffer=False,
                 interval=1,
                 data_info=None,
                 use_apex_amp=False,
                 loss_name='loss_path_regular'):
        super().__init__()
        self.loss_weight = loss_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.decay = decay
        self.pl_batch_size = pl_batch_size
        self.sync_mean_buffer = sync_mean_buffer
        self.interval = interval
        self.data_info = data_info
        self.use_apex_amp = use_apex_amp
        self._loss_name = loss_name

        self.register_buffer('mean_path_length', torch.tensor(0.))

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``gen_path_regularizer``.
        """
        if self.interval > 1:
            assert self.data_info is not None
        # use data_info to build computational path
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')
            if self.interval > 1 and outputs_dict[
                    'iteration'] % self.interval != 0:
                return None
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            kwargs.update(
                dict(
                    weight=self.loss_weight,
                    mean_path_length=self.mean_path_length,
                    pl_batch_shrink=self.pl_batch_shrink,
                    decay=self.decay,
                    use_apex_amp=self.use_apex_amp,
                    pl_batch_size=self.pl_batch_size,
                    sync_mean_buffer=self.sync_mean_buffer))
            path_penalty, self.mean_path_length, _ = gen_path_regularizer(
                **kwargs)

            return path_penalty
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return gen_path_regularizer(
                *args, weight=self.loss_weight, **kwargs)

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


def third_party_net_loss(net, weight=1.0, **kwargs):
    return net(**kwargs) * weight


@MODULES.register_module()
class FaceIdLoss(nn.Module):
    """Face similarity loss. Generally this loss is used to keep the id
    consistency of the input face image and output face image.

    In this loss, we may need to provide ``gt``, ``pred`` and ``x``. Thus,
    an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:
        data_info = dict(
            gt='real_imgs',
            pred='fake_imgs')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        facenet (dict, optional): Config dict for facenet. Defaults to
            dict(type='ArcFace', ir_se50_weights=None, device='cuda').
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_id'.
    """

    def __init__(self,
                 loss_weight=1.0,
                 data_info=None,
                 facenet=dict(
                     type='ArcFace', ir_se50_weights=None, device='cuda'),
                 loss_name='loss_id'):

        super(FaceIdLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info
        self.net = build_module(facenet)
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``third_party_net_loss``.
        """
        # use data_info to build computational path
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            kwargs.update(dict(weight=self.loss_weight))
            return third_party_net_loss(self.net, *args, **kwargs)
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return third_party_net_loss(
                self.net, *args, weight=self.loss_weight, **kwargs)

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class CLIPLossModel(torch.nn.Module):
    """Wrapped clip model to calculate clip loss.

    Ref: https://github.com/orpatashnik/StyleCLIP/blob/main/criteria/clip_loss.py # noqa

    Args:
        in_size (int, optional): Input image size. Defaults to 1024.
        scale_factor (int, optional): Unsampling factor. Defaults to 7.
        pool_size (int, optional): Pooling output size. Defaults to 224.
        clip_type (str, optional): A model name listed by
            `clip.available_models()`, or the path to a model checkpoint
            containing the state_dict. For more details, you can refer to
            https://github.com/openai/CLIP/blob/573315e83f07b53a61ff5098757e8fc885f1703e/clip/clip.py#L91 # noqa
            Defaults to 'ViT-B/32'.
        device (str, optional): Model device. Defaults to 'cuda'.
    """

    def __init__(self,
                 in_size=1024,
                 scale_factor=7,
                 pool_size=224,
                 clip_type='ViT-B/32',
                 device='cuda'):
        super(CLIPLossModel, self).__init__()
        try:
            import clip
        except ImportError:
            raise 'To use clip loss, openai clip need to be installed first'
        self.model, self.preprocess = clip.load(clip_type, device=device)
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor)
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=(scale_factor * in_size // pool_size))

    def forward(self, image=None, text=None):
        """Forward function."""
        assert image is not None
        assert text is not None
        image = self.avg_pool(self.upsample(image))
        loss = 1 - self.model(image, text)[0] / 100
        return loss


@MODULES.register_module()
class CLIPLoss(nn.Module):
    """Clip loss. In styleclip, this loss is used to optimize the latent code
    to generate image that match the text.

    In this loss, we may need to provide ``image``, ``text``. Thus,
    an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:
        data_info = dict(
            image='fake_imgs',
            text='descriptions')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        clip_model (dict, optional): Kwargs for clip loss model. Defaults to
            dict().
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_clip'.
    """

    def __init__(self,
                 loss_weight=1.0,
                 data_info=None,
                 clip_model=dict(),
                 loss_name='loss_clip'):

        super(CLIPLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info
        self.net = CLIPLossModel(**clip_model)
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``third_party_net_loss``.
        """
        # use data_info to build computational path
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            kwargs.update(dict(weight=self.loss_weight))
            return third_party_net_loss(self.net, *args, **kwargs)
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return third_party_net_loss(
                self.net, *args, weight=self.loss_weight, **kwargs)

    @staticmethod
    def loss_name():
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return 'clip_loss'


class PerceptualVGG(nn.Module):
    """VGG network used in calculating perceptual loss.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): According to the name in this list,
            forward function will return the corresponding features. This
            list contains the name each layer in `vgg.feature`. An example
            of this list is ['4', '10'].
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image.
            Importantly, the input feature must in the range [0, 1].
            Default: True.
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 pretrained='torchvision://vgg19'):
        super().__init__()
        if pretrained.startswith('torchvision://'):
            assert vgg_type in pretrained
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # get vgg model and load pretrained vgg weight
        # remove _vgg from attributes to avoid `find_unused_parameters` bug
        _vgg = getattr(vgg, vgg_type)()
        self.init_weights(_vgg, pretrained)
        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        # only borrow layers that will be used from _vgg to avoid unused params
        self.vgg_layers = _vgg.features[:num_layers]

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for v in self.vgg_layers.parameters():
            v.requires_grad = False

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output

    def init_weights(self, model, pretrained):
        """Init weights.

        Args:
            model (nn.Module): Models to be inited.
            pretrained (str): Path for pretrained weights.
        """
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)


@MODULES.register_module()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    .. code-block:: python
        :caption: Code from StaticUnconditionalGAN, train_step
        :linenos:

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size)

    But in this loss, we may need to provide ``pred`` and ``target`` as input.
    Thus, an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            pred='fake_imgs',
            target='real_imgs',
            layer_weights={
                     '4': 1.,
                     '9': 1.,
                     '18': 1.},
            )

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_mse'.
        layers_weights (dict): The weight for each layer of vgg feature for
            perceptual loss. Here is an example: {'4': 1., '9': 1., '18': 1.},
            which means the 5th, 10th and 18th feature layer will be
            extracted with weight 1.0 in calculating losses. Defaults to
            '{'4': 1., '9': 1., '18': 1.}'.
        layers_weights_style (dict): The weight for each layer of vgg feature
            for style loss. If set to 'None', the weights are set equal to
            the weights for perceptual loss. Default: None.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
        split_style_loss (bool): Whether return a separate style loss item.
            Options are True and False. Default: False
    """

    def __init__(self,
                 data_info=None,
                 loss_name='loss_perceptual',
                 layer_weights={
                     '4': 1.,
                     '9': 1.,
                     '18': 1.
                 },
                 layer_weights_style=None,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 perceptual_weight=1.0,
                 style_weight=1.0,
                 norm_img=True,
                 pretrained='torchvision://vgg19',
                 criterion='l1',
                 split_style_loss=False):
        super().__init__()
        self.data_info = data_info
        self._loss_name = loss_name
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.layer_weights_style = layer_weights_style
        self.split_style_loss = split_style_loss

        self.vgg = PerceptualVGG(
            layer_name_list=list(self.layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            pretrained=pretrained)

        if self.layer_weights_style is not None and \
                self.layer_weights_style != self.layer_weights:
            self.vgg_style = PerceptualVGG(
                layer_name_list=list(self.layer_weights_style.keys()),
                vgg_type=vgg_type,
                use_input_norm=use_input_norm,
                pretrained=pretrained)
        else:
            self.layer_weights_style = self.layer_weights
            self.vgg_style = None

        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = l1_loss
        elif criterion == 'mse':
            self.criterion = mse_loss
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, *args, **kwargs):
        """Forward function. If ``self.data_info`` is not ``None``, a
        dictionary containing all of the data and necessary modules should be
        passed into this function. If this dictionary is given as a non-keyword
        argument, it should be offered as the first argument. If you are using
        keyword argument, please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function, ``mse_loss``.

        Args:
            pred (Tensor): Input tensor with shape (n, c, h, w).
            target (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # use data_info to build computational path
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            return self.perceptual_loss(**kwargs)
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return self.perceptual_loss(*args, **kwargs)

    def perceptual_loss(self, pred, target):
        if self.norm_img:
            pred = (pred + 1.) * 0.5
            target = (target + 1.) * 0.5
        # extract vgg features
        pred_features = self.vgg(pred)
        target_features = self.vgg(target.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in pred_features.keys():
                percep_loss += self.criterion(
                    pred_features[k],
                    target_features[k],
                    weight=self.layer_weights[k])
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = 0.

        # calculate style loss
        if self.style_weight > 0:
            if self.vgg_style is not None:
                pred_features = self.vgg_style(pred)
                target_features = self.vgg_style(target.detach())

            style_loss = 0
            for k in pred_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(pred_features[k]),
                    self._gram_mat(
                        target_features[k])) * self.layer_weights_style[k]
            style_loss *= self.style_weight
        else:
            style_loss = 0.

        if self.split_style_loss:
            return percep_loss, style_loss
        else:
            return percep_loss + style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
