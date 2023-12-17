import torch
from pathlib import Path
from model.models import BERTTest
from transformers import BertTokenizer
from utils.dataprocess import SingleSentenceDataset
def evaluate(model, data, device, save_model_path=None):
    if save_model_path is not None:
        model.load_state_dict(torch.load(save_model_path))
    model.eval()
    input_ids = data[0]['input_ids'].to(device)
    mask = data[0]['attn_mask'].to(device)
    with torch.no_grad():
        active_logits = model(input_ids, mask=mask)
        probabilities = torch.nn.functional.softmax(active_logits.float(), dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=-1)
    print(f"输入字符：{input_ids}")
    print(f"输入掩码：{mask}")
    print(f"输出标签：{predicted_labels}")
def test():
    bert_path = "/home/cv/train/ner/bert-base-chinese"
    save_model_path = Path(__file__).parent /'weights/100.pt'
    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = "昔 日 津 门 足 球 的 雄 风 ， 成 为 天 津 足 坛 上 下 内 外 到 处 议 论 的 话 题 。"
    input = input.replace(" ","") 
    print(len(input))
    test_Dataset = SingleSentenceDataset(sentence=input,tokenizer=tokenizer,seq_len=40)
    model = BERTTest(bert_path=bert_path,label_count=31).to(device)
    evaluate(model, test_Dataset, device, save_model_path=save_model_path)
if __name__ == "__main__":
    test()
