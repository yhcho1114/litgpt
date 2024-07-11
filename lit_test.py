import fire
import sacrebleu
from datasets import load_dataset
from litgpt import LLM
import json
from tqdm import tqdm  # Import tqdm

def load_70B_outputjson(iter):
    # load /workspace/SAIL-Server-Assisted-Inferences-of-LLM/70B_output.json
    with open('/home/nxc/yhcho/sail/litgpt2/litgpt/70B_output.json') as f:
        data = json.load(f)
    return data[iter]


def main(
    model_name: str = "google/gemma-2-9b",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    num_samples: int = 1000,  # Number of sentences to translate
):
    # Load the translation dataset
    dataset = load_dataset("wmt20_mlqe_task1", "en-de")
    test_data = dataset["test"]

    # Initialize the LLM model
    llm = LLM.load(model_name)

    translations_output = []
    prompts = [
        f"Translate the following German sentence into English: \n"
        """
The Sultan appoints judges, and can grant pardons and commute sentences. => Der Sultan ernennt Richter und kann Begnadigungen und Pendelstrafen gewähren.
Antisemitism in modern Ukraine Antisemitism and Special Relativity => Antisemitismus in der modernen Ukraine Antisemitismus und besondere Relativität
Morales continued his feud with Buddy Rose, defeating him by disqualification. => Morales setzte seine Fehde mit Buddy Rose fort und besiegte ihn durch Disqualifikation.
American Maury Tripp attended the Jamboree from Saratoga, California. => Der Amerikaner Maury Tripp besuchte das Jamboree aus Saratoga, Kalifornien.
He bowled a series of bouncers at Viv Richards at Brisbane and claimed 3/77 and 5/92 in the Third Test at Melbourne. => Er boomte eine Reihe von Bouncern bei Viv Richards in Brisbane und behauptete 3 / 77 und 5 / 92 im dritten Test in Melbourne.
"""
        f"{sentence['translation']['en']} => "
        for sentence in test_data.select(range(num_samples))
    ]

    # Translate each sentence and calculate BLEU scores
    total_bleu_score = 0
    probabilities = []
    lengths = []
    pbar = tqdm(enumerate(prompts), total=len(prompts))
    total_ratio = 0
    for i, prompt in pbar:  # Use tqdm here
        #print(prompt)
        large_generated = load_70B_outputjson(i)
        result = llm.generate(
            prompt,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        #print(result)
        translation_raw = result
        total_ratio += len(translation_raw) / len(prompt)
    
        # Remove leading newlines and spaces
        translation_cleaned = translation_raw.lstrip()
        
        # Extract the first line after cleaning
        translation = translation_cleaned.split('\n')[0].strip()
        translation = translation.replace('\'', '')  # Remove apostrophes

        reference = test_data[i]['translation']['de']
        bleu_score = sacrebleu.corpus_bleu([translation], [[reference]]).score
        total_bleu_score += bleu_score
        average_bleu_score = total_bleu_score/(i+1)
        pbar.set_description(f"Processing {i+1}/{num_samples}, BLEU: {average_bleu_score:.2f}, Ratio: {total_ratio/(i+1):.2f}")

        # Append probability in dictionary of probabilities
        
        probabilities.append({"translation": translation, "reference": reference, "bleu_score": bleu_score})

    # Calculate average BLEU score
    output_file_name = f'litgpt_{model_name.replace("/", "_")}_output.json'
    with open(output_file_name, 'w', encoding='utf-8') as f:
        json.dump(probabilities, f, ensure_ascii=False, indent=4)

    # Calculate average BLEU score
    average_bleu_score = total_bleu_score / num_samples
    print(f"Average BLEU Score: {average_bleu_score}")
    # save output as 70B_output.json


if __name__ == "__main__":
    fire.Fire(main)