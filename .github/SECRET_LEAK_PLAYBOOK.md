# Secret Leak Remediation Playbook

Ce playbook standardise la reponse a un leak detecte par gitleaks.
Objectif: rotation rapide des secrets, purge de l historique Git, verification, puis prevention.

## 1) Triage immediat (0 a 15 min)
1. Ouvrir un incident securite.
2. Identifier le type de secret et son fournisseur (API key, token, password, webhook, etc.).
3. Evaluer l exposition:
   - branche/fichier/commit concernes
   - repo public ou prive
   - secret present dans l historique ou seulement dans le dernier commit
4. Ne jamais copier la valeur du secret dans les issues, PR ou commentaires.

## 2) Containment et rotation (15 a 60 min)
1. Revoquer ou desactiver immediatement le secret expose.
2. Regenerer un nouveau secret.
3. Mettre a jour les systemes consommateurs (CI, runtime, vault, environnements).
4. Verifier que les appels avec l ancien secret echouent et que le nouveau fonctionne.

## 3) Purge Git (si secret present dans l historique)
Utiliser un clone miroir pour nettoyer toutes les refs.

```bash
git clone --mirror git@github.com:OWNER/REPO.git
cd REPO.git
```

### Cas A: supprimer un fichier complet qui contenait le secret
```bash
git filter-repo --path chemin/du/fichier --invert-paths
```

### Cas B: remplacer une valeur dans tout l historique
Creer un fichier `replacements.txt` avec le format:

```text
regex:ANCIEN_SECRET==>***REMOVED***
```

Puis executer:

```bash
git filter-repo --replace-text replacements.txt
```

Publier l historique nettoye:

```bash
git push --force --tags origin 'refs/heads/*'
```

Informer les contributeurs qu ils doivent resynchroniser leur clone local.

## 4) Verification post-remediation
1. Relancer le workflow gitleaks manuellement (`workflow_dispatch`).
2. Verifier qu aucun secret n est detecte.
3. Verifier que les applications/CI fonctionnent avec les nouveaux secrets.
4. Fermer l incident avec:
   - cause racine
   - impact
   - actions correctives

## 5) Prevention
1. Stocker les secrets uniquement dans GitHub Secrets ou un secret manager.
2. Interdire les secrets en clair dans `.env*` versionnes.
3. Ajouter des patterns de detection custom dans `gitleaks.toml` si necessaire.
4. Former les contributeurs sur la procedure de rotation.

## Checklist rapide
- [ ] Secret revoque
- [ ] Secret remplace
- [ ] Historique purge (si necessaire)
- [ ] Scan gitleaks relance et vert
- [ ] Incident documente et cloture
