import json
import torch
from torch.utils.data import Dataset
import random

def build_repair_dataset_from_joint(joint_path, re_label_id2str, neg_per_pos=1, seed=42):
    """
    Builds an entity-pair RE dataset with all positives and up to `neg_per_pos` negatives per positive per example.
    """
    random.seed(seed)
    data = json.load(open(joint_path))
    out_rows = []
    all_labels = set()
    for example in data:
        tokens = example["tokens"]
        entities = example["entities"]
        relations = example.get("relations", [])
        rel_lookup = {(rel["head"], rel["tail"]): rel["type"] for rel in relations}
        pos_pairs = []
        neg_pairs = []
        # Collect all pairs
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i == j:
                    continue
                label = rel_lookup.get((i, j), "NO_REL")
                if label == "NO_REL":
                    neg_pairs.append((i, j, label))
                else:
                    pos_pairs.append((i, j, label))
        # Add all positives
        for i, j, label in pos_pairs:
            out_rows.append({
                "tokens": tokens,
                "entity1": entities[i],
                "entity2": entities[j],
                "label": label
            })
            all_labels.add(label)
        # Sample negatives (at most neg_per_pos * num_pos per example)
        num_negs = min(len(neg_pairs), neg_per_pos * max(1, len(pos_pairs)))
        sampled_neg_pairs = random.sample(neg_pairs, num_negs) if num_negs > 0 else []
        for i, j, label in sampled_neg_pairs:
            out_rows.append({
                "tokens": tokens,
                "entity1": entities[i],
                "entity2": entities[j],
                "label": label
            })
            all_labels.add(label)
    print(f"Built {len(out_rows)} examples ({len(all_labels)} relation types).")
    return out_rows


class REDataset(Dataset):
    def __init__(self, path, tokenizer, label2id, max_length=256, devel=False):
        re_label2id = {"NO_REL": 0, "Relation": 1}
        id2rel = [rel for rel, _ in sorted(re_label2id.items(), key=lambda x: x[1])]
        if devel:
            self.data = build_repair_dataset_from_joint(path, id2rel, neg_per_pos=3, seed=93)
        else:
            self.data = build_repair_dataset_from_joint(path, id2rel)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        tokens = ex["tokens"]
        e1 = ex["entity1"]
        e2 = ex["entity2"]

        # Mark entity tokens with unique special tokens
        marked_tokens = tokens[:]
        # Use entity index insertion order to avoid marker overlap
        s1, e1_end = e1["start"], e1["end"] - 1
        s2, e2_end = e2["start"], e2["end"] - 1
        # Handle possible entity overlap
        if s1 < s2:
            marked_tokens[s1] = "[E1]" + marked_tokens[s1]
            marked_tokens[e1_end] = marked_tokens[e1_end] + "[/E1]"
            marked_tokens[s2] = "[E2]" + marked_tokens[s2]
            marked_tokens[e2_end] = marked_tokens[e2_end] + "[/E2]"
        else:
            marked_tokens[s2] = "[E2]" + marked_tokens[s2]
            marked_tokens[e2_end] = marked_tokens[e2_end] + "[/E2]"
            marked_tokens[s1] = "[E1]" + marked_tokens[s1]
            marked_tokens[e1_end] = marked_tokens[e1_end] + "[/E1]"

        inputs = self.tokenizer(
            marked_tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        label = self.label2id.get(ex["label"], 0)
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(label)
        return item
