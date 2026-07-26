# -*- coding: utf-8 -*-
import json
import numpy as np

SNIPPETS = [
    {"id": "hadith-bukhari", "genre": "Hadith",  "book": "صحيح البخاري",
     "text": "إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى، فمن كانت هجرته إلى دنيا يصيبها أو إلى امرأة ينكحها فهجرته إلى ما هاجر إليه"},
    {"id": "tafsir-zad-almasir", "genre": "Tafsir", "book": "زاد المسير في علم التفسير",
     "text": "أقبل المسلمون على كتاب ربهم وكلام خالقهم دراسة وحفظا وعملا، وألفوا في علومه كتبا ومؤلفات عديدة في التفسير والقراءات واستنباط الأحكام"},
    {"id": "fiqh-mudawwana-1", "genre": "Fiqh", "book": "المدونة (مالكي)",
     "text": "قال سحنون: قلت لعبد الرحمن بن القاسم: أرأيت الوضوء أكان مالك يوقت فيه واحدة أو اثنتين أو ثلاثا؟ قال: لا إلا ما أسبغ، ولم يكن مالك يوقت"},
    {"id": "fiqh-mudawwana-2", "genre": "Fiqh", "book": "المدونة (مالكي)",
     "text": "وقال مالك: لا بأس بعرق البرذون والبغل والحمار، وإن ولغ الكلب في إناء فيه لبن فلا بأس بأن يؤكل ذلك اللبن"},
    {"id": "aqidah-tahawiyya", "genre": "Aqidah", "book": "شرح العقيدة الطحاوية",
     "text": "والعقيدة هي مأخوذة من العقد وهو الربط، وسميت عقيدة؛ لأن الإنسان يجزم ويعتقد في نفسه"},
    {"id": "philology-dhurrumma", "genre": "Philology", "book": "ديوان ذي الرمة (شرح)",
     "text": "السفعة: ما خالف لون الأرض، وهو يضرب إلى السواد. والنكباء: ريح تجيء منحرفة بين ريحين"},
    {"id": "quran-16-67", "genre": "Quran", "book": "القرآن الكريم (١٦:٦٧)",
     "text": "وَمِنْ ثَمَرَاتِ النَّخِيلِ وَالْأَعْنَابِ تَتَّخِذُونَ مِنْهُ سَكَرًا وَرِزْقًا حَسَنًا إِنَّ فِي ذَلِكَ لَآيَةً لِقَوْمٍ يَعْقِلُونَ"},
    {"id": "quran-2-25", "genre": "Quran", "book": "القرآن الكريم (٢:٢٥)",
     "text": "وَبَشِّرِ الَّذِينَ آمَنُوا وَعَمِلُوا الصَّالِحَاتِ أَنَّ لَهُمْ جَنَّاتٍ تَجْرِي مِنْ تَحْتِهَا الْأَنْهَارُ كُلَّمَا رُزِقُوا مِنْهَا مِنْ ثَمَرَةٍ رِزْقًا قَالُوا هَذَا الَّذِي رُزِقْنَا مِنْ قَبْلُ"},
]

def main():
    from sentence_transformers import SentenceTransformer

    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name)

    texts = ["passage: " + s["text"] for s in SNIPPETS]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings)

    # cosine similarity matrix (already normalized -> dot product)
    sim = embeddings @ embeddings.T

    ids = [s["id"] for s in SNIPPETS]
    result = {
        "model": model_name,
        "dimensions": int(embeddings.shape[1]),
        "snippets": SNIPPETS,
        "similarity_matrix": sim.tolist(),
        "ids_order": ids,
    }
    with open("embed_demo_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    np.save("embed_demo_vectors.npy", embeddings)
    print("DONE. dims=", embeddings.shape)
    print("Similarity matrix:")
    for i, row in enumerate(sim):
        print(ids[i], [round(x, 3) for x in row])

if __name__ == "__main__":
    main()
