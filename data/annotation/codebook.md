# Codebook d'annotation — Intentions de dialogue (Cornell)

> Guide pour annoter les répliques de cinéma (Cornell Movie-Dialogs) avec une
> **macro-classe d'intention**. Objectif : créer un gold standard in-domain pour
> mesurer le transfert SWDA → Cornell (Option B), notamment le F1 sur ORDRE.

---

## 0. Règles d'or

1. **Étiquette la RÉPLIQUE CIBLE** (`texte`), pas le contexte. Le contexte (`contexte_avant`, `contexte_apres`) sert **uniquement à désambiguïser**, jamais à être étiqueté.
2. **Une seule étiquette par réplique** : la **fonction principale** de l'énoncé. Si une réplique fait deux choses, choisis l'acte dominant.
3. **Tu travailles en aveugle.** Tu ne connais ni le genre des personnages, ni le film, ni l'année. C'est voulu : ne devine pas, n'extrapole pas à partir d'indices supposés.
4. **En cas de doute réel et persistant** → `AMBIGU`. Ne force pas un choix faux. Mais `AMBIGU` est un dernier recours, pas une facilité.
5. **Note ta confiance** (`confiance_1_3`) : `1` = incertain, `2` = plutôt sûr, `3` = certain.
6. **Champ `notes`** : libre, pour signaler un cas limite ou justifier un choix difficile.

---

## 1. Les 9 macro-classes (+ AMBIGU)

### STATEMENT — assertion factuelle / descriptive
Affirmation neutre, information, narration, description. **Pas** d'évaluation subjective forte.
- *« I went to the store yesterday. »*
- *« It's raining outside. »*
- *« He works at the bank downtown. »*
> Frontière avec OPINION : si c'est un fait/une description → STATEMENT ; si c'est un jugement/une croyance → OPINION.

### OPINION — évaluation, croyance, jugement
Point de vue subjectif, appréciation, conviction. Souvent *I think / I believe / I feel / should*.
- *« I think this is a terrible idea. »*
- *« She's the best in the business. »*
- *« You should really apologize to him. »*

### QUESTION — demande d'information ou de confirmation
Toute interrogation : ouverte (wh-), fermée (yes/no), rhétorique, ou **déclarative à valeur de question**.
- *« What time is it? »*
- *« Do you want some coffee? »*
- *« You're leaving already? »* (forme déclarative, fonction question)

### ORDRE — directive d'action (action-directive)
Énoncé qui **demande à l'interlocuteur de faire quelque chose**. Impératif adressé, requête d'action.
- *« Get in the car. »*  *« Close the door, please. »*  *« Don't touch that! »*
- Requêtes polies d'action : *« Could you pass the salt? »* → **ORDRE** (fonction directive).
> ⚠️ **Pièges (mémoriser) :**
> - *« Let me… / Let's… »* = **PAS** ORDRE (hortatif 1ère personne, action du locuteur lui-même) → souvent STATEMENT ou AUTRE_DIALOGUE.
> - Directives **indirectes** (*« Why don't you… », « You might want to… »*) : ambiguës avec OPINION/QUESTION. Si la fonction dominante est de **faire agir** → ORDRE (confiance basse) ; si c'est surtout un avis → OPINION. Mets `confiance_1_3 = 1` et une note.

### ACCORD — accord, acceptation, réponse affirmative
Approbation explicite, acceptation d'une proposition, *yes-answer* à une question.
- *« Yeah, absolutely. »*  *« That's exactly right. »*  *« Yes, I'll do it. »*
> Frontière avec BACKCHANNEL : voir ci-dessous.

### DESACCORD — désaccord, rejet, réponse négative
Contestation, refus, *no-answer*, minimisation/contradiction.
- *« No, that's not true. »*  *« I disagree completely. »*  *« Absolutely not. »*

### POLITESSE — formules conventionnelles
Excuses, remerciements, salutations d'ouverture, formules de clôture.
- *« I'm so sorry. »*  *« Thank you very much. »*  *« Hello, how are you? »*  *« Goodbye, take care. »*

### BACKCHANNEL — signal d'écoute (continuer)
Brève réaction qui **n'ajoute pas de contenu** : montre qu'on écoute, qu'on suit. Souvent pendant que l'autre parle.
- *« Uh-huh. »*  *« Mm-hmm. »*  *« Right. »*  *« I see. »*
> ⚠️ **ACCORD vs BACKCHANNEL** — le piège le plus fréquent. Le **contexte tranche** :
> - *« Yeah »* en **réponse à une question** ou pour valider une proposition → **ACCORD**.
> - *« Yeah »* comme **simple continuateur** pendant que l'autre développe → **BACKCHANNEL**.

### AUTRE_DIALOGUE — actes de gestion du dialogue
Complétions collaboratives, tag-questions, maintien du tour de parole (*hold*), citations rapportées, hedges (atténuateurs).
- *« …and then he just left, right? »* (tag)
- *« Well, you know, I mean… »* (hold / hedge)
- *« She said “I'll be back”. »* (citation)

### AMBIGU — indécidable
Même **avec le contexte**, l'acte reste véritablement indéterminé (texte trop tronqué, sens dépendant d'éléments absents). Dernier recours.

---

## 2. Procédure pas à pas

1. Lis `contexte_avant` → `texte` → `contexte_apres`.
2. Demande-toi : **quelle est la fonction principale de `texte` dans l'échange ?**
3. Choisis **une** macro-classe (ou `AMBIGU`).
4. Renseigne `confiance_1_3`.
5. Si cas limite / hésitation entre deux classes → note-le dans `notes`.

---

## 3. Accord inter-annotateur & adjudication

- **3 annotateurs** (A, B, C). Un **bloc de recouvrement (~200 répliques)** est annoté par les 3.
- Mesure d'accord : **Krippendorff's α (nominal)** en principal (gère 3 juges, AMBIGU, manquants), **Fleiss' κ** en complément.
- `AMBIGU` est une **catégorie à part entière** dans le calcul de l'accord (ne pas la jeter — elle reflète la difficulté réelle).
- **Gold du recouvrement** : vote majoritaire (2/3) ; désaccords 3-voies ou impliquant AMBIGU → **adjudication par discussion**.
- Hors recouvrement : annotation simple, le label unique fait gold (qualité bornée par l'α mesuré).

---

## 4. Rappels d'hygiène

- Ne cherche pas à deviner le genre du personnage : c'est la variable étudiée, l'introduire biaiserait toute l'analyse.
- Le contexte brut peut contenir un prénom ou un *« sir / ma'am »* : **ignore-le** comme indice de genre, sers-t'en seulement pour le sens de l'acte.
- Pas de prédiction du modèle sous les yeux : ton jugement doit être indépendant.
