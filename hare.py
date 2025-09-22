import torch
import torch.nn.functional as F
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# =======================
# Model Loader
# =======================
class HAREEvaluator:
    def __init__(self, device=None):
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    
        alignment_model_name = "knowlab-research/HARE-alignment"
        ner_model_name = "knowlab-research/HARE-NER"
        re_model_name = "knowlab-research/HARE-RE"
    
        # Load alignment model
        self.eval_tokenizer = AutoTokenizer.from_pretrained(alignment_model_name)
        self.eval_model = AutoModel.from_pretrained(alignment_model_name).eval().to(self.device)
    
        # Load NER pipeline
        self.pipe = pipeline(
            "ner", model=ner_model_name, aggregation_strategy="average",
            device=self.device if torch.cuda.is_available() else -1
        )
    
        # Load RE model
        self.re_tokenizer = AutoTokenizer.from_pretrained(re_model_name)
        self.re_model = AutoModelForSequenceClassification.from_pretrained(re_model_name).to(self.device).eval()
        self.id2label = {0: "NO_REL", 1: "Relation"}

    # -----------------------
    # Internal utilities
    # -----------------------
    def chunk_text(self, text, tokenizer, chunk_size=450):
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens), chunk_size):
            yield tokens[i : i + chunk_size]

    def process_text(self, text):
        entities = []
        for chunk in self.chunk_text(text, tokenizer=self.pipe.tokenizer):
            chunk_text_str = self.pipe.tokenizer.decode(chunk, skip_special_tokens=True)
            entities.extend(self.pipe(chunk_text_str))
        return entities

    def process_embedding(self, entities):
        if len(entities) == 0:
            return torch.tensor([])
        embeds_word = torch.tensor([]).to(self.device)
        with torch.no_grad():
            encoded = self.eval_tokenizer(
                entities, truncation=True, padding=True,
                return_tensors="pt", max_length=30
            ).to(self.device)
            embeds_word = torch.cat(
                (embeds_word.to("cpu"), self.eval_model(**encoded).last_hidden_state[:, 0, :].to("cpu")),
                dim=0,
            )
        return embeds_word

    def get_score(self, gt_entities, pred_entities, threshold=0.7):
        gt_entities = [e["word"] for e in gt_entities if e["score"] > threshold]
        pred_entities = [e["word"] for e in pred_entities if e["score"] > threshold]
        if len(gt_entities) == 0 or len(pred_entities) == 0:
            return 0.0

        gt_embeddings = self.process_embedding(gt_entities)
        pred_embeddings = self.process_embedding(pred_entities)

        gt_similarities = [F.cosine_similarity(emb, pred_embeddings, dim=1).max() for emb in gt_embeddings]
        pred_similarities = [F.cosine_similarity(gt_embeddings, emb, dim=1).max() for emb in pred_embeddings]

        recall = np.average(gt_similarities)
        precision = np.average(pred_similarities)

        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def predict_relations(self, text, entities, threshold=0.7):
        results = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i == j:
                    continue
                s1, e1_end = e1["start"], e1["end"]
                s2, e2_end = e2["start"], e2["end"]
                marked_text = (
                    text[:s1]
                    + "[E1]" + text[s1:e1_end] + "[/E1]"
                    + text[e1_end:s2]
                    + "[E2]" + text[s2:e2_end] + "[/E2]"
                    + text[e2_end:]
                )
                inputs = self.re_tokenizer(
                    marked_text, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    logits = self.re_model(**inputs).logits
                    pred = logits.argmax(dim=-1).item()
                    label = self.id2label[pred]
                    score = torch.softmax(logits, dim=-1)[0, pred].item()
                if score > threshold:
                    results.append({"head": e1, "tail": e2, "pred_label": label, "score": score})
        return results

    def get_relation_f1(self, gt_relations, pred_relations):
        gt_rel_set = {
            (rel["head"]["word"], rel["tail"]["word"], rel["pred_label"])
            for rel in gt_relations if rel["pred_label"] != "NO_REL"
        }
        pred_rel_set = {
            (rel["head"]["word"], rel["tail"]["word"], rel["pred_label"])
            for rel in pred_relations if rel["pred_label"] != "NO_REL"
        }
        correct = len(pred_rel_set & gt_rel_set)
        precision = correct / (len(pred_rel_set) + 1e-9)
        recall = correct / (len(gt_rel_set) + 1e-9)
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    # -----------------------
    # Main evaluation
    # -----------------------
    def evaluate(self, pred_text, gt_text, alpha=0.5):
        pred_entities = self.process_text(pred_text)
        gt_entities = self.process_text(gt_text)
        ner_f1 = self.get_score(gt_entities, pred_entities)
        pred_relations = self.predict_relations(pred_text, pred_entities)
        gt_relations = self.predict_relations(gt_text, gt_entities)
        re_f1 = self.get_relation_f1(gt_relations, pred_relations)
        harescore = alpha * ner_f1 + (1 - alpha) * re_f1
        return {"harescore": harescore}


# =======================
# CLI
# =======================
# pred_text="**Histological Findings:**  \n- Presence of invasive neoplastic glands with irregular architecture.  \n- Desmoplastic stromal reaction surrounding the neoplastic glands.  \n- Tumor cells exhibit nuclear pleomorphism and hyperchromasia.  \n- Evidence of necrosis and mitotic activity.  \n\n**Microscopic Description:**  \nThe slide shows a tumor composed of atypical glandular structures infiltrating the surrounding stroma. The glands are irregularly shaped and lined by cells with enlarged, hyperchromatic nuclei and prominent nucleoli. The stroma is desmoplastic, and areas of necrosis are noted. Mitotic figures are frequent.  \n\n**Final Diagnosis:**  \nAdenocarcinoma."
# gt_text="The pathology report describes a case of endometrial endometrioid adenocarcinoma with atypical hyperplasia in a patient who underwent a total vaginal hysterectomy. The tumor is diffusely involving the endometrial cavity (>90%) and abuts the serosa without invading beyond it. The maximum depth of myometrial invasion is 4.5 cm in a 4.5 cm thick wall, and lymphatic/vascular invasion is present. The tumor is poorly differentiated and has a predominant (>80%) solid growth pattern with extensive areas of necrosis and focal pseudorosette/pseudoglandular formation. The cervix shows no diagnostic abnormality, and the specimen margins are not involved. Lymph node excisions from the left and right obturator, left and right pelvis, left aortic, and right aortic regions were negative for malignancy (0/2, 0/6, 0/4, 0/1, and 0/1, respectively), with foci of endosalpingiosis found in one of the left obturator nodes."
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run HARE evaluation")
    parser.add_argument("--pred", type=str, required=True, help="Predicted text")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth text")
    args = parser.parse_args()

    evaluator = HAREEvaluator()
    results = evaluator.evaluate(args.pred, args.gt)
    print("Results:", results)


