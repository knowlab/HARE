# HARE

<<<<<<< HEAD
[**Paper to be released soon**]() | 
=======
[**Paper accepted at EMNLP2025**](https://arxiv.org/pdf/2509.16326) | 
>>>>>>> 44965b9 (added training script)
[**🤗 Model and Dataset**](https://huggingface.co/collections/knowlab-research/hare-68d145c54cdc98743baafe1c) | 

HARE is Histopathology Automated Report Evaluation. 

It is a novel entity and relation centric framework, composed of a benchmark dataset, a named entity recognition (NER) model, a relation extraction (RE) model, and a novel metric, which prioritizes clinically relevant content by aligning critical histopathology entities and relations between reference and generated reports.


# How to use it

## Set up Environment

conda create -n hare python=3.10 -y
pip install -r requirements.txt

## Import and run evaluator
```
from hare import HAREEvaluator

evaluator = HAREEvaluator()

pred_text="**Histological Findings:**  \n- Presence of invasive neoplastic glands with irregular architecture.  \n- Desmoplastic stromal reaction surrounding the neoplastic glands.  \n- Tumor cells exhibit nuclear pleomorphism and hyperchromasia.  \n- Evidence of necrosis and mitotic activity.  \n\n**Microscopic Description:**  \nThe slide shows a tumor composed of atypical glandular structures infiltrating the surrounding stroma. The glands are irregularly shaped and lined by cells with enlarged, hyperchromatic nuclei and prominent nucleoli. The stroma is desmoplastic, and areas of necrosis are noted. Mitotic figures are frequent.  \n\n**Final Diagnosis:**  \nAdenocarcinoma."

gt_text="The pathology report describes a case of endometrial endometrioid adenocarcinoma with atypical hyperplasia in a patient who underwent a total vaginal hysterectomy. The tumor is diffusely involving the endometrial cavity (>90%) and abuts the serosa without invading beyond it. The maximum depth of myometrial invasion is 4.5 cm in a 4.5 cm thick wall, and lymphatic/vascular invasion is present. The tumor is poorly differentiated and has a predominant (>80%) solid growth pattern with extensive areas of necrosis and focal pseudorosette/pseudoglandular formation. The cervix shows no diagnostic abnormality, and the specimen margins are not involved. Lymph node excisions from the left and right obturator, left and right pelvis, left aortic, and right aortic regions were negative for malignancy (0/2, 0/6, 0/4, 0/1, and 0/1, respectively), with foci of endosalpingiosis found in one of the left obturator nodes."

results = evaluator.evaluate(pred_text, gt_text)
print(results)
```

# Dataset
The NER and RE dataset are released.
The histopathologist 600 reports annotation is released `human_evaluation_annotation.tsv'

## Example for running HARE on 600 reports
> Please refer to example.ipynb

# Hardware
The code was tested with A5000 GPU 24GB memory.

<<<<<<< HEAD
=======
## Running NER training
```
CUDA_VISIBLE_DEVICES=0 python run_ner.py \
    --model_name_or_path UFNLP/gatortronS \
    --train_file train_ner.json \
    --validation_file test_ner.json \
    --output_dir ./gatortronS-NER \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-05 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --save_steps 0.5 \
    --logging_steps 0.5
```
## Running RE training
```
CUDA_VISIBLE_DEVICES=1 python re_trainer.py \
    --model_name UFNLP/gatortronS \
    --output_dir ./gatortronS-RE
```
>>>>>>> 44965b9 (added training script)
