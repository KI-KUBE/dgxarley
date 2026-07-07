# SOPS HOWTO — Secrets verschlüsseln & mit Ansible deployen

Praktischer Leitfaden: SOPS einrichten, Secrets verschlüsselt ins Git legen und
beim Ansible-Deploy transparent entschlüsseln. Als Key-Backend nutzen wir
**age** (modern, keine GPG-Keyring-Schmerzen).

> **SOPS** = *Secrets OPerationS*, Projekt der CNCF (früher Mozilla).
> Es verschlüsselt **nur die Werte** in YAML/JSON/ENV/INI-Dateien, nicht die
> Keys. Dadurch bleibt die Datei diff- und mergebar — man sieht *welcher*
> Schlüssel sich ändert, nur nicht der Wert.

---

## 0. Überblick — wie es zusammenspielt

```
                 ┌─────────────┐   verschlüsselt    ┌──────────────┐
   du (Editor) ─▶│ sops edit    │ ─────────────────▶ │ vault.yml    │──▶ git
                 └─────────────┘   (age public key)  │ (im Klartext │
                                                     │  nur Werte   │
                                                     │  chiffriert) │
   ansible ◀── entschlüsselt ◀── sops (age secret) ◀─┴──────────────┘
```

- **Public key** → verschlüsselt. Darf ins Repo (`.sops.yaml`).
- **Secret key** (`keys.txt`) → entschlüsselt. Bleibt **lokal / im Secret-Store**, nie ins Git.
- Wer deployen können soll, braucht den Secret Key. Wer nur schreiben will,
  kommt mit dem Public Key aus.

---

## 1. Installation

**sops** (empfohlen via Homebrew, hält sich selbst aktuell):

```bash
brew install sops
sops --version    # >= 3.9
```

Alternativ das offizielle Release-Binary:

```bash
curl -L -o ~/.local/bin/sops \
  https://github.com/getsops/sops/releases/latest/download/sops-v3.13.2.linux.amd64
chmod +x ~/.local/bin/sops
```

**age** (Key-Erzeugung + Krypto-Backend):

```bash
# Debian/Ubuntu:
sudo apt install age
# oder brew install age
age --version
```

---

## 2. age-Key erzeugen (einmalig pro Person/Maschine)

```bash
mkdir -p ~/.config/sops/age
age-keygen -o ~/.config/sops/age/keys.txt
chmod 600 ~/.config/sops/age/keys.txt
```

Ausgabe (Beispiel):

```
Public key: changeme_age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8j
```

- `~/.config/sops/age/keys.txt` ist der **Default-Pfad**, den sops automatisch
  findet — kein Setzen von Umgebungsvariablen nötig.
- Den **Public key** (Zeile oben, oder `age-keygen -y ~/.config/sops/age/keys.txt`)
  brauchst du gleich für `.sops.yaml`.
- **Backup des `keys.txt`** an einen sicheren Ort (Passwortmanager / Vault).
  Verlierst du ihn, sind alle nur-mit-diesem-Key verschlüsselten Secrets weg.

Public key jederzeit wieder ausgeben:

```bash
age-keygen -y ~/.config/sops/age/keys.txt
```

---

## 3. `.sops.yaml` im Repo-Root anlegen

Diese Datei sagt sops **welche Dateien mit welchen Keys** verschlüsselt werden.
Sie gehört **ins Git** (enthält nur Public Keys). Beispiel für ein
Ansible-Layout:

```yaml
# .sops.yaml
keys:
  # >>> jede Person, die entschlüsseln darf, mit ihrem age-PUBLIC-key <<<
  - &thiess age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8j
  - &ci     age1cizzz...ci-deploy-key...zzz

creation_rules:
  # Alle Ansible-Vault-Dateien
  - path_regex: (group_vars|host_vars)/.*/vault\.ya?ml$
    key_groups:
      - age:
          - *thiess
          - *ci

  # Freistehende Secret-Dateien
  - path_regex: secrets/.*\.(ya?ml|json|env)$
    key_groups:
      - age:
          - *thiess
          - *ci

  # Fallback für alles andere, das man explizit verschlüsselt
  - key_groups:
      - age:
          - *thiess
```

Merke:
- **Nur die erste passende `creation_rule`** greift (Reihenfolge = Priorität).
- YAML-Anker (`&thiess` / `*thiess`) vermeiden Copy-Paste der langen Keys.
- Nur die hier gelisteten Keys können später entschlüsseln. Neuen Menschen
  hinzufügen = Key ergänzen + `sops updatekeys` (siehe §8).

---

## 3b. SSH-Keys als Recipients (statt eigener age-Keys)

Wenn das Team schon **SSH-ed25519-Keys** hat (`~/.ssh/id_ed25519`, in
GitHub/GitLab hinterlegt, auf den Deploy-Hosts vorhanden), kann man die direkt
als Recipients nehmen — niemand muss extra einen age-Key erzeugen und
verwalten. Es gibt zwei Wege.

> Verifiziert mit **sops 3.13.2** auf diesem System. Nativer SSH-Support gibt es
> ab sops ~3.9. Es funktionieren **nur `ssh-ed25519` und `ssh-rsa`** — keine
> ecdsa-/`sk-`-Keys.

### Weg 1 — SSH-Pubkey direkt (nativ, kein Extra-Tool) ✅ empfohlen

age *und* sops verstehen SSH-Keys direkt. Man trägt die **komplette
Pubkey-Zeile** als age-Recipient in `.sops.yaml` ein — **nicht** einen
`age1…`-String:

```yaml
# .sops.yaml
keys:
  - &thiess_ssh ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB9xtjTmS6oCcWrgN/8MBrZqcuwmnx9E... thiess@hyperion
  - &ci_ssh     ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA...ci-deploy...                    ci@runner

creation_rules:
  - path_regex: (group_vars|host_vars)/.*/vault\.ya?ml$
    key_groups:
      - age:
          - *thiess_ssh
          - *ci_ssh
```

Verschlüsseln passiert dann wie gewohnt (`sops group_vars/prod/vault.yml`). In
der verschlüsselten Datei steht als Recipient die SSH-Zeile, z. B.
`recipient: ssh-ed25519 AAAA...`.

**Entschlüsseln** — sops findet den passenden SSH-**Privkey** so:

- **automatisch** unter `~/.ssh/id_ed25519` bzw. `~/.ssh/id_rsa` — *keine
  Env-Variable nötig* (getestet).
- oder explizit, wenn der Key woanders liegt:

  ```bash
  export SOPS_AGE_SSH_PRIVATE_KEY_FILE=/pfad/zu/id_ed25519
  sops decrypt group_vars/prod/vault.yml
  ```

Fallstricke:
- **`SOPS_AGE_KEY_FILE` funktioniert NICHT mit einem SSH-Key** (die Variable
  erwartet age-Format `AGE-SECRET-KEY-1…`). Für SSH den Default-Pfad oder
  `SOPS_AGE_SSH_PRIVATE_KEY_FILE` nutzen.
- **Passphrase-geschützte** SSH-Keys fragt age interaktiv ab — für
  automatisierte Deploys/CI daher einen **passwortlosen Deploy-Key** verwenden
  (ssh-agent hilft hier nicht, age spricht nicht mit dem Agent).

### Weg 2 — SSH → age umrechnen mit `ssh-to-age`

Manche Setups (z. B. **sops-nix**) oder Tooling, das nur `age1…`-Recipients
kennt, wollen einheitliche age-Keys. `ssh-to-age` rechnet ed25519-SSH-Keys
**deterministisch** in age-Keys um.

```bash
brew install ssh-to-age            # bei dir aktuell nicht installiert

# SSH-Pubkey  -> age-Recipient (age1...) für .sops.yaml:
ssh-to-age -i ~/.ssh/id_ed25519.pub
curl -s https://github.com/DEIN_USER.keys | ssh-to-age     # direkt aus GitHub

# SSH-Privkey -> age-Identity (zum Entschlüsseln) an den Default-Pfad:
ssh-to-age -private-key -i ~/.ssh/id_ed25519 -o ~/.config/sops/age/keys.txt
```

Den resultierenden `age1…`-Recipient dann wie in §3 in `.sops.yaml` eintragen;
die `keys.txt` am Default-Pfad wird beim Entschlüsseln automatisch gefunden.

### Wann welcher Weg

| | Weg 1 (SSH direkt) | Weg 2 (`ssh-to-age`) |
|---|---|---|
| Extra-Tool | keins | `ssh-to-age` |
| Recipient in `.sops.yaml` | `ssh-ed25519 AAAA…` | `age1…` |
| Single source of truth | der SSH-Key selbst | umgerechneter age-Key |
| Gut wenn | einfachster Start, SSH-Keys eh vorhanden | einheitliche `age1…`-Recipients (sops-nix, Mischbetrieb mit reinen age-Keys) |

**Empfehlung:** Weg 1, außer du hast einen konkreten Grund für einheitliche
`age1…`-Recipients.

### Deploy mit SSH-Keys

Auf dem Deploy-Runner ist nichts Zusätzliches nötig — der SSH-Privkey liegt auf
Ops-Maschinen meist ohnehin. Ansible-Deploy dann exakt wie in §6; sops greift
automatisch auf `~/.ssh/id_ed25519` zu.

CI mit SSH-Deploy-Key:

```yaml
- name: SSH-Deploy-Key bereitstellen
  run: |
    mkdir -p ~/.ssh
    echo "${{ secrets.DEPLOY_SSH_KEY }}" > ~/.ssh/id_ed25519
    chmod 600 ~/.ssh/id_ed25519
- name: Deploy
  run: ansible-playbook -i inventory/prod site.yml   # sops nutzt ~/.ssh/id_ed25519 automatisch
```

---

## 4. Secrets verschlüsseln & bearbeiten

**Neue Datei anlegen / bestehende im Editor bearbeiten** (öffnet `$EDITOR`,
verschlüsselt beim Speichern automatisch nach den `.sops.yaml`-Regeln):

```bash
sops group_vars/prod/vault.yml
```

Du tippst Klartext-YAML:

```yaml
db_password: super-secret-123
api_token: ghp_xxxxxxxxxxxx
```

Nach dem Speichern liegt auf der Platte / im Git nur noch:

```yaml
db_password: ENC[AES256_GCM,data:9f3k...,tag:...]
api_token:  ENC[AES256_GCM,data:a1b2...,tag:...]
sops:
    age:
        - recipient: age1ql3z...
          enc: |
            -----BEGIN AGE ENCRYPTED FILE-----
            ...
    lastmodified: "2026-07-07T..."
    mac: ENC[...]
```

Weitere nützliche Kommandos:

```bash
# Klartext-Datei nachträglich in-place verschlüsseln
sops encrypt --in-place secrets/api.yml

# Entschlüsselt nach stdout ausgeben (nicht speichern) — für schnelles Prüfen
sops decrypt group_vars/prod/vault.yml

# Nur einen einzelnen Wert setzen/holen ohne Editor
sops set   group_vars/prod/vault.yml '["db_password"]' '"neues-pw"'
sops decrypt --extract '["db_password"]' group_vars/prod/vault.yml
```

---

## 5. Ansible-Integration

Der saubere Weg ist die **`community.sops` Collection** — damit entschlüsselt
Ansible `vault.yml`-Dateien automatisch beim Laden, ohne dass du irgendwo
manuell `sops decrypt` aufrufst.

### 5.1 Einrichtung (einmalig)

```bash
ansible-galaxy collection install community.sops
```

`requirements.yml` fürs Repo (damit CI/Kollegen es reproduzieren):

```yaml
# requirements.yml
collections:
  - name: community.sops
```

### 5.2 Variante A — `vars_files` / `include_vars` (explizit)

In einem Playbook lädst du eine verschlüsselte Datei via Lookup-Plugin:

```yaml
- hosts: dbservers
  vars:
    vault: "{{ lookup('community.sops.sops', 'group_vars/prod/vault.yml') | from_yaml }}"
  tasks:
    - name: DB-Passwort setzen
      ansible.builtin.debug:
        msg: "pw ist {{ vault.db_password }}"
```

### 5.3 Variante B — Vars-Plugin (automatisch, empfohlen)

Aktivier das **vars-Plugin**, dann werden `group_vars`/`host_vars`, die
sops-verschlüsselt sind, ganz normal wie unverschlüsselte Vars geladen —
transparent, ohne Lookup:

```ini
# ansible.cfg
[defaults]
vars_plugins_enabled = host_group_vars,community.sops.sops
```

Danach reicht ganz normales Ansible:

```yaml
- hosts: dbservers
  tasks:
    - ansible.builtin.debug:
        msg: "pw ist {{ db_password }}"   # kommt aus group_vars/prod/vault.yml
```

Ansible ruft im Hintergrund sops auf; sops findet deinen age-Key unter
`~/.config/sops/age/keys.txt` und entschlüsselt on-the-fly.

---

## 6. Deploy-Workflow

**Voraussetzung auf der Deploy-Maschine (Laptop oder CI-Runner):**
`sops` + `age` installiert, und der **age-Secret-Key** verfügbar.

```bash
# 1. Collections holen
ansible-galaxy collection install -r requirements.yml

# 2. sicherstellen, dass der Key da ist
ls -l ~/.config/sops/age/keys.txt

# 3. deployen — Secrets werden zur Laufzeit entschlüsselt, nie im Klartext geschrieben
ansible-playbook -i inventory/prod site.yml
```

Schneller Vorab-Check, dass Entschlüsselung klappt, bevor du das ganze
Playbook fährst:

```bash
sops decrypt group_vars/prod/vault.yml | head
# oder ansible-seitig:
ansible -i inventory/prod dbservers -m debug -a "var=db_password"
```

### 6.1 CI/CD (GitLab / GitHub Actions)

Den age-Secret-Key als **CI-Secret** hinterlegen und zur Laufzeit in den
Default-Pfad schreiben — nie ins Repo:

```yaml
# GitHub Actions Beispiel
- name: age-Key bereitstellen
  run: |
    mkdir -p ~/.config/sops/age
    echo "${{ secrets.SOPS_AGE_KEY }}" > ~/.config/sops/age/keys.txt
    chmod 600 ~/.config/sops/age/keys.txt

- name: Deploy
  run: |
    ansible-galaxy collection install -r requirements.yml
    ansible-playbook -i inventory/prod site.yml
```

Alternativ zum Datei-Pfad geht auch die Env-Var `SOPS_AGE_KEY` (Inhalt des
Keys direkt) — praktisch für Container:

```bash
export SOPS_AGE_KEY="AGE-SECRET-KEY-1..."
ansible-playbook -i inventory/prod site.yml
```

---

## 7. Git-Integration (schöne Diffs)

Damit `git diff` bei verschlüsselten Dateien den **Klartext-Diff** zeigt statt
Chiffrat-Müll:

```bash
# .gitattributes
group_vars/**/vault.yml diff=sops
host_vars/**/vault.yml  diff=sops
secrets/**              diff=sops
```

```bash
# einmalig lokal konfigurieren
git config diff.sops.textconv "sops decrypt"
```

Jetzt zeigt `git diff` die entschlüsselten Werte — lokal, nur für Leute mit Key.

> Optional: ein `pre-commit`-Hook, der verhindert, dass versehentlich eine
> **unverschlüsselte** `vault.yml` committet wird (prüft, ob das `sops:`-Feld
> in der Datei vorhanden ist).

---

## 8. Key-Management & Rotation

**Neue Person / neuen CI-Key hinzufügen:**

1. Deren age-**Public key** in `.sops.yaml` unter `keys:` + in die passende
   `creation_rule` eintragen.
2. Bestehende Dateien mit den neuen Keys neu verschlüsseln:

   ```bash
   sops updatekeys group_vars/prod/vault.yml
   # oder alle auf einmal:
   find . -name 'vault.yml' -exec sops updatekeys -y {} \;
   ```

   `updatekeys` verschlüsselt **nur den Datenschlüssel neu**, nicht deinen
   Klartext — schnell und diff-freundlich.

**Person entfernen:** Public key aus `.sops.yaml` löschen → `sops updatekeys`.
Beachte: Wer die alte Git-History hat, kann alte Versionen weiter
entschlüsseln → bei echtem Leak die **Secrets selbst rotieren** (Passwörter neu).

**Data-Key rotieren** (frischer Verschlüsselungs-Schlüssel, z. B. nach Verdacht):

```bash
sops rotate --in-place group_vars/prod/vault.yml
```

---

## 9. Fehlerbehebung

| Symptom | Ursache / Fix |
|---|---|
| `no matching creation rules found` | Pfad matcht keine `creation_rule` in `.sops.yaml`. Regex prüfen. |
| `failed to get the data key ... no identity matched` | Dein Public key steht nicht in der Datei. Von jemandem mit Zugang `sops updatekeys` laufen lassen. |
| Ansible sieht Vars nicht | `community.sops` nicht installiert **oder** `vars_plugins_enabled` in `ansible.cfg` fehlt. |
| sops findet Key nicht | `keys.txt` nicht unter `~/.config/sops/age/` oder `SOPS_AGE_KEY(_FILE)` nicht gesetzt. |
| SSH-Key wird nicht akzeptiert | `SOPS_AGE_KEY_FILE` zeigt auf SSH-Key → falsch. `SOPS_AGE_SSH_PRIVATE_KEY_FILE` nutzen oder Key nach `~/.ssh/id_ed25519` legen. Nur ed25519/rsa unterstützt. |
| CI: `age: no identity` | CI-Secret `SOPS_AGE_KEY` fehlt/leer oder Datei nicht `chmod 600`. |

---

## 10. Merkzettel (die 6 Kommandos, die man täglich braucht)

```bash
sops secrets/foo.yml                       # anlegen/bearbeiten (verschlüsselt beim Speichern)
sops decrypt secrets/foo.yml               # Klartext ansehen
sops encrypt --in-place plain.yml          # Klartextdatei verschlüsseln
sops updatekeys secrets/foo.yml            # Keys aus .sops.yaml neu anwenden
sops rotate --in-place secrets/foo.yml     # Data-Key rotieren
ansible-playbook -i inventory/prod site.yml  # deploy (entschlüsselt automatisch)
```

---

### Anhang: SOPS vs. git-crypt (dieses Repo nutzt aktuell git-crypt)

Dieses Repo verschlüsselt Vault-Dateien heute mit **git-crypt** (siehe
`.gitattributes`). Kurzer Vergleich, falls ein Umstieg zur Debatte steht:

| | git-crypt | SOPS |
|---|---|---|
| Granularität | ganze Datei (binär im Git) | einzelne **Werte**, Keys bleiben lesbar |
| Diff/Merge | Datei = Blob, kaum mergebar | zeilenweise diffbar (welcher Key sich ändert) |
| Key-Backend | GPG / symmetrisch | age, GPG, KMS, Vault, … |
| Zugriff granular | pro Repo | pro Datei-Regex + Key-Gruppen |
| Ansible | Datei muss ausgecheckt entschlüsselt sein | `community.sops` entschlüsselt on-the-fly |

**Nicht beides gleichzeitig auf dieselben Dateien** anwenden. Ein Umstieg würde
bedeuten: git-crypt-Filter für `vault.yml` entfernen, Dateien entschlüsseln,
mit `sops` neu verschlüsseln, `.gitattributes`/`.sops.yaml` anpassen — bewusst
als eigener Schritt, nicht nebenbei.
```
