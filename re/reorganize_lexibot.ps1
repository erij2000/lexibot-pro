param(
    [Parameter(Mandatory=$false)]
    [Switch]$DryRun,

    [Parameter(Mandatory=$false)]
    [Switch]$Backup
)

$BaseDir = "LEXIBOT_PRO/backend"
$ProblemFiles = @(
    "$BaseDir/app/calendar.py",
    "$BaseDir/app/chatbot.py",
    "$BaseDir/auth/auth.py",
    "$BaseDir/schemas/security.py"
)
$ProblemFolders = @(
    "$BaseDir/app",
    "$BaseDir/auth"
)
$MissingInit = @(
    "$BaseDir/api/routers",
    "$BaseDir/api/dependencies",
    "$BaseDir/core",
    "$BaseDir/database",
    "$BaseDir/models",
    "$BaseDir/schemas"
)

function Log-Action {
    param(
        [string]$Message,
        [string]$Color = "Green"
    )
    Write-Host "$Message" -ForegroundColor $Color
}

function Run-Reorganization {
    Log-Action "--- ÉTAPE 1 : Analyse de la structure actuelle ---"
    $ProblemsDetected = 0

    # Vérification des doublons
    Log-Action "📂 Vérification des doublons et des anciens dossiers..."
    foreach ($File in $ProblemFiles) {
        if (Test-Path $File) {
            Log-Action "   ❌ Doublon détecté : $File" -Color Yellow
            $ProblemsDetected++
        }
    }
    foreach ($Folder in $ProblemFolders) {
        if (Test-Path $Folder) {
            Log-Action "   ❌ Ancien dossier détecté : $Folder" -Color Yellow
            $ProblemsDetected++
        }
    }

    # Vérification des __init__.py manquants
    Log-Action "📝 Vérification des __init__.py manquants..."
    $MissingInitCount = 0
    foreach ($InitDir in $MissingInit) {
        if (-not (Test-Path "$InitDir/__init__.py")) {
            Log-Action "   ⚠️  Manquant : $InitDir/__init__.py" -Color Cyan
            $MissingInitCount++
            $ProblemsDetected++
        }
    }

    Log-Action "📊 Résumé de l'analyse :"
    Log-Action "   Total de problèmes détectés : $ProblemsDetected" -Color Yellow

    if ($DryRun) {
        Log-Action "Mode simulation terminé. Aucune modification n'a été effectuée." -Color Green
        return
    }

    if ($ProblemsDetected -eq 0) {
        Log-Action "Aucun problème majeur détecté. La structure est propre." -Color Green
        return
    }

    Log-Action "--- ÉTAPE 2 : Suppression des doublons ---"
    
    # 1. Suppression des fichiers doublons
    foreach ($File in $ProblemFiles) {
        if (Test-Path $File) {
            Remove-Item -Path $File -Force
            Log-Action "✅ Supprimé : $File"
        }
    }

    # 2. Suppression des dossiers problématiques
    foreach ($Folder in $ProblemFolders) {
        if (Test-Path $Folder) {
            Remove-Item -Path $Folder -Recurse -Force
            Log-Action "✅ Supprimé (Dossier) : $Folder"
        }
    }

    Log-Action "--- ÉTAPE 3 : Création des __init__.py manquants ---"
    
    # Création des __init__.py
    foreach ($InitDir in $MissingInit) {
        $InitFile = "$InitDir/__init__.py"
        if (-not (Test-Path $InitFile)) {
            New-Item -Path $InitFile -ItemType File -Force | Out-Null
            Log-Action "✅ Créé : $InitFile"
        }
    }
    
    Log-Action "--- Réorganisation TERMINÉE ---" -Color Green
}

# --- Logique d'exécution ---

if ($Backup) {
    $BackupDirName = "LEXIBOT_PRO_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    Log-Action "Création d'une sauvegarde de LEXIBOT_PRO dans $BackupDirName..." -Color Blue
    Copy-Item -Path "LEXIBOT_PRO" -Destination $BackupDirName -Recurse -Force
    Log-Action "Sauvegarde créée avec succès." -Color Blue
}

if (-not $DryRun) {
    Write-Host "
⚠️  ATTENTION ! Cette opération va modifier votre projet." -ForegroundColor Red
    Write-Host "Continuer ? (O/N): " -NoNewline
    $Confirmation = Read-Host
    if ($Confirmation -ne "O") {
        Log-Action "Opération annulée par l'utilisateur." -Color Yellow
        exit
    }
}

Run-Reorganization
