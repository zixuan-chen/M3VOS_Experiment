from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, get_hidden_state



model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
# image_file = "https://llava-vl.github.io/static/images/view.jpg"
image_file = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem_LLM/test.png"


args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "return_dict": True
})()
# args.pop("cache_position"


hidden_state = get_hidden_state(args)
# eval_model(args)
print(hidden_state)

