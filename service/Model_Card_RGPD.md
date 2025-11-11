# Model Card – Conformité RGPD (Digital Social Score)

**Projet** : toxicity-project  
**GCP Project ID** : toxicity-projet2  
**Version modèle** : 2.1.0

## 1. Finalité et usage
- Détection de propos toxiques pour assister la modération.  
- Hors périmètre : décisions automatisées à effet légal, profilage étendu.

## 2. Base légale (RGPD)
- Consentement (si applicable).  
- Intérêt légitime (art. 6(1)(f)) pour la prévention des abus.

## 3. Catégories de données
- Entrées : texte libre (peut contenir DCP).  
- Métadonnées minimales : horodatage, IP LB (sécurité).  
- Catégories particulières : non visées intentionnellement.

## 4. Minimisation & anonymisation
- Anonymisation amont (regex + spaCy PERSON/EMAIL/IP/ADDRESS/USERNAME/CREDIT_CARD).  
- Les payloads d’inférence ne sont pas stockés par défaut.  


## 5. Localisation & sécurité
- GCP (europe-west1) – GKE, Artifact Registry, Logging, Secret Manager.  
- TLS en transit, IAM, isolation par conteneurs, CI/CD.  
- Pas d’exposition de scores bruts ; décision binaire “toxic / non toxic”.

## 6. Droits des personnes (RGPD)
- Accès, rectification, effacement, opposition : via DPO/Support.  
- Portabilité : si conservation de données personnelles (non prévue par défaut).

## 8. Gouvernance
- Revue à chaque changement majeur.  
- Journal des changements (extrait) :  
  - 2.1.0 – Remplacement des scores par label binaire ; durcissement prétraitement.  
  - 2.0.0 – Migration GKE + CI/CD.

## 9. Contacts
- Propriétaire produit : tallawilly@icloud.com  
- Tech lead : tallawilly@icloud.com  


**Dépôt** : https://github.com/Willy772/toxicity-project
