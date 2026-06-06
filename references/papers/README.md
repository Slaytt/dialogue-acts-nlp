# Références — papiers NLP utiles au projet

5 PDF téléchargés depuis ACL Anthology. Classement par utilité pour notre Bloc 3 (annotation Cornell) puis pour la rédaction du papier (RQ2/RQ3).

## Pour le protocole d'annotation (Bloc 3 — priorité)

### `stolcke_2000_dialogue_act_modeling.pdf`
Stolcke et al. (2000), *Computational Linguistics* 26(3), 339–374.
**Le papier-source du SWDA-DAMSL.** À extraire :
- Schéma d'annotation 42 classes → comment ils ont défini les classes
- κ inter-annotateurs (rapporté ~0.84) → benchmark à viser
- Protocole de désaccord, double annotation, cas limites
- Justification du mapping vers macro-classes (réduction du schéma)

### `artstein_poesio_2008_intercoder_agreement.pdf`
Artstein & Poesio (2008), *Computational Linguistics* 34(4), 555–596.
**LA référence sur l'IAA.** À extraire :
- Cohen's κ vs Krippendorff's α → lequel choisir
- Seuils acceptables (κ > 0.6 substantial, > 0.8 perfect)
- Comment reporter l'IAA proprement dans la section méthodo

## Pour le contexte film/genre (RQ2/RQ3 + intro/related work)

### `schofield_mehr_2016_gender_film_dialogue.pdf`
Schofield & Mehr (2016), *CLfL workshop*.
**Cornell + analyse genre, papier court.** À extraire :
- Quelle stratification ont-ils utilisée
- Quelles features linguistiques distinguent les genres
- Ce qu'ils n'ont PAS fait (analyse dyadique = notre différenciateur C2)

### `ramakrishna_2017_movie_characters_linguistic.pdf`
Ramakrishna et al. (2017), *ACL*.
**Analyse de scénarios de films, niveau personnage.** À extraire :
- Méthodo psycholinguistique (LIWC etc.)
- Comment ils contrôlent les confondants (genre du film, époque, rôle)
- Limites qu'ils mentionnent (utiles pour notre discussion)

## Pour le bias transfer (cadrage cousine)

### `bertsch_2022_gender_bias_transfer_film.pdf`
Bertsch et al. (2022), *GeBNLP workshop*.

⚠️ **Correction sur ma description initiale** : ce papier porte sur le **transfert de biais en pré-entraînement BERT sur OpenSubtitles** (mesure SEAT, sentiment analysis), **pas** sur le transfert d'un classifieur SWDA vers Cornell. Donc moins directement utile pour notre protocole d'annotation que ce que j'ai annoncé. Il reste pertinent pour :
- Le cadrage "biais dans données de films"
- Les métriques de biais (SEAT) si on veut élargir
- À discuter avec la cousine pour la section interprétative C3

## Lecture proposée (ordre)

1. **Stolcke 2000** §6 (Annotation methodology) — 5 pages
2. **Artstein & Poesio 2008** §1-3 (kappa, alpha, choix) — ~15 pages
3. **Schofield & Mehr 2016** complet — 6 pages, vite lu
4. (Plus tard) Ramakrishna 2017 et Bertsch 2022
