

def build_model_vost(args):
    # backbone
    if args.model.name.backbone == "backbone":
        from .backbone import build_backbone
        backbone = build_backbone(args)
    elif args.model.name.backbone == "backbone_sep_mask":
        from .backbone import build_backbone
        backbone = {'image': build_backbone(args), 'mask': build_backbone(args)}
    else:
        raise ValueError(f"Invalid backbone name: {args.model.name.backbone}")

    # transformer
    if args.model.name.transformer == "transformer_vost_multi_scale_memory_last_layer_too_ms":
        from .transformer_vost_multi_scale_memory_last_layer_too_ms import build_transformer
        transformer = build_transformer(args)
    else:
        raise ValueError(f"Invalid transformer name: {args.model.name.transformer}")

    # tubedetr
    if args.model.name.tubedetr == "tubedetr_vost_multi_scale_memory_ms":
        from .tubedetr_vost_multi_scale_memory_ms import build_tubedetr
    else:
        raise ValueError(f"Invalid tubedetr name: {args.model.name.tubedetr}")

    return build_tubedetr(args, backbone, transformer)


def build_model_static(args):
    # backbone
    if args.model.name.backbone == "backbone":
        from .backbone import build_backbone
        backbone = build_backbone(args)
    else:
        raise ValueError(f"Invalid backbone name: {args.model.name.backbone}")

    # transformer
    if args.model.name.transformer == "transformer_static_multi_scale":
        from .transformer_static_multi_scale import build_transformer
        transformer = build_transformer(args)
    elif args.model.name.transformer == "transformer_static_multi_scale_memory":
        from .transformer_static_multi_scale_memory import build_transformer
        transformer = build_transformer(args)
    elif args.model.name.transformer == "transformer_static_single_scale_memory":
        from .transformer_static_single_scale_memory import build_transformer
        transformer = build_transformer(args)
    elif args.model.name.transformer == "transformer_static_multi_scale_memory_last_layer_too_ms":
        from .transformer_static_multi_scale_memory_last_layer_too_ms import build_transformer
        transformer = build_transformer(args)
    else:
        raise ValueError(f"Invalid transformer name: {args.model.name.transformer}")

    # tubedetr
    if args.model.name.tubedetr == "tubedetr_static_multi_scale":
        from .tubedetr_static_multi_scale import build_tubedetr
    elif args.model.name.tubedetr == "tubedetr_static_multi_scale_memory":
        from .tubedetr_static_multi_scale_memory import build_tubedetr
    elif args.model.name.tubedetr == "tubedetr_static_single_scale_memory":
        from .tubedetr_static_single_scale_memory import build_tubedetr
    elif args.model.name.tubedetr == "tubedetr_static_multi_scale_memory_ms":
        from .tubedetr_static_multi_scale_memory_ms import build_tubedetr
    else:
        raise ValueError(f"Invalid tubedetr name: {args.model.name.tubedetr}")

    return build_tubedetr(args, backbone, transformer)
