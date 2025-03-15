import fvcore.nn.weight_init as weight_init

from torch import nn
from torch.nn import functional as F
from detectron2.layers import Conv2d, ShapeSpec, get_norm


INPUT_SHAPE = {
    'res2': ShapeSpec(channels=256, height=None, width=None, stride=4),
    'res3': ShapeSpec(channels=512, height=None, width=None, stride=8),
    'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16),
    'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)
}


# This is a modified FPN decoder.
class BasePixelDecoder(nn.Module):
    def __init__(self, args):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        task_name = args.tasks.names[0]

        conv_dim = args.model[task_name].pixel_decoder.conv_dim
        mask_dim = args.model[task_name].pixel_decoder.mask_dim
        norm = args.model[task_name].pixel_decoder.norm

        input_shape = {
            k: v for k, v in INPUT_SHAPE.items() if k in args.model[task_name].pixel_decoder.in_features
        }

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            # if idx == len(self.in_features) - 1:
            #     output_norm = get_norm(norm, conv_dim)
            #     output_conv = Conv2d(
            #         in_channels,
            #         conv_dim,
            #         kernel_size=3,
            #         stride=1,
            #         padding=1,
            #         bias=use_bias,
            #         norm=output_norm,
            #         activation=F.relu,
            #     )
            #     weight_init.c2_xavier_fill(output_conv)
            #     self.add_module("layer_{}".format(idx + 1), output_conv)

            #     lateral_convs.append(None)
            #     output_convs.append(output_conv)
            # else:
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales

    def forward_features(self, features, img_only_memory_reshaped):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        # import ipdb; ipdb.set_trace()
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            # if lateral_conv is None:
            if idx == 0:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + img_only_memory_reshaped[0]
                y = output_conv(y)
            elif idx == 1:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest") + img_only_memory_reshaped[1]
                y = output_conv(y)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), None, multi_scale_features

    def forward(self, features, targets=None):
        return self.forward_features(features)


class PixelDecoderUsingTransformerEncodedFeatures(BasePixelDecoder):
    def __init__(self, args):
        super().__init__(args)

        task_name = args.tasks.names[0]
        conv_dim = args.model[task_name].pixel_decoder.conv_dim
        norm = args.model[task_name].pixel_decoder.norm

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv


def build_pixel_decoder(args):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    assert len(args.tasks.names) == 1
    task_name = args.tasks.names[0]
    name = args.model[task_name].pixel_decoder.name
    if name == "BasePixelDecoder":
        model = BasePixelDecoder(args)
    elif name == "PixelDecoderUsingTransformerEncodedFeatures":
        model = PixelDecoderUsingTransformerEncodedFeatures(args)
    else:
        raise ValueError()
    return model
