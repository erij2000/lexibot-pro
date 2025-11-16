param(
    [switch]$Verbose,
    [switch]$FixImports
)

# --- Couleurs ---
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Cyan = "Cyan"

Write-Host "`n╔════════════════════════════════╗" -ForegroundColor $Cyan
Write-Host "║   DIAGNOSTIC DES IMPORTS     ║" -ForegroundColor $Cyan
Write-Host "╚════════════════════════════════╝`n" -ForegroundColor $Cyan

$errors = @()
$fixes = @()
$tempScriptPath = ".\temp_ast_reader.py"

# --------------------------
# Step 1: Python environment
# --------------------------
# On part du répertoire racine du projet (où backend est situé)
# Pour une meilleure portabilité, nous nous assurons que nous sommes dans le répertoire parent
$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "`nChecking Python Environment..." -ForegroundColor $Cyan

$python = "python"
if (Test-Path ".\venv\Scripts\python.exe") {
    $python = ".\venv\Scripts\python.exe"
} elseif (Test-Path ".\venv\bin\python") {
    $python = ".\venv\bin\python"
}

try {
    $version = & $python --version 2>&1
    Write-Host "   ✅ Python: $version" -ForegroundColor $Green
} catch {
    Write-Host "   ❌ Python non trouvé! Assurez-vous qu'il est dans PATH ou venv est activé." -ForegroundColor $Red
    exit 1
}

# --------------------------
# Step 2: Create temporary AST reader script
# --------------------------
# Ce script Python sera exécuté pour chaque fichier pour extraire les imports.
# Nous utilisons un script externe pour éviter les erreurs de parsing PowerShell.
$astScriptContent = @"
import ast
import sys

def get_imports(file_path):
    try:
        with open(file_path, encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)
        
        imps = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    # Ajout d'une ligne pour s'assurer que l'import fonctionne
                    imps.append(f'import {n.name}') 
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                names = ','.join([n.name for n in node.names])
                # Ajout d'une ligne pour s'assurer que l'import fonctionne
                imps.append(f'from {module} import {names}') 
        print('|'.join(imps))
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to parse AST for {file_path}: {e}", file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        get_imports(sys.argv[1])
"@

Set-Content -Path $tempScriptPath -Value $astScriptContent -Encoding UTF8

# --------------------------
# Step 3: Collect Python files
# --------------------------
# Recherche des fichiers dans le dossier 'backend'
$pythonFiles = Get-ChildItem -Recurse -Filter "*.py" -Path ".\backend" |
    Where-Object { $_.FullName -notmatch "(__pycache__|venv|migrations|temp_ast_reader)" }

Write-Host "`nFichiers Python trouvés: $($pythonFiles.Count)" -ForegroundColor $Cyan

# --------------------------
# Step 4: Test imports
# --------------------------
foreach ($file in $pythonFiles) {
    if ($Verbose) { Write-Host "`nProcessing $($file.FullName)" -ForegroundColor $Yellow }

    # 4a. Utiliser le script temporaire pour obtenir les imports
    $importResult = & $python $tempScriptPath $file.FullName 2>&1
    
    # Vérifie si l'analyse AST a généré des erreurs
    if ($importResult -match "ERROR:") {
        $errors += "Erreur d'analyse AST dans $($file.FullName): $importResult"
        Write-Host "   ❌ $($file.Name) (Erreur d'analyse)" -ForegroundColor $Red
        continue
    }

    $imports = $importResult -split '\|' | Where-Object { $_ -ne "" }
    if ($imports.Count -eq 0) { 
        if ($Verbose) { Write-Host "   🟡 $($file.Name) : Pas d'imports trouvés" -ForegroundColor $Yellow }
        continue 
    }

    # 4b. Teste tous les imports en un seul appel Python
    $testScript = ($imports | ForEach-Object { "$_; " }) + "print('SUCCESS')"
    $testResult = & $python -c $testScript 2>&1

    if ($testResult -notmatch "SUCCESS") {
        # L'erreur est la dernière ligne avant 'SUCCESS'
        $errorLine = $testResult -split "`n" | Select-Object -Last 1
        $errors += "Import échoué dans $($file.FullName): $errorLine"
        Write-Host "   ❌ $($file.Name) : Imports échoués" -ForegroundColor $Red
    } elseif ($Verbose) {
        Write-Host "   ✅ $($file.Name) : Imports OK" -ForegroundColor $Green
    }

    # 4c. Optional: fix relative imports (Gardé pour référence future)
    # (Le fix pour 'auth.auth_config' est devenu obsolète avec la nouvelle architecture,
    # nous le laissons donc comme placeholder si vous en avez besoin plus tard)
    if ($FixImports) {
        $content = Get-Content $file.FullName -Raw
        $patterns = @{
            "from database.config import" = "from backend.database.session import"
            "from auth.auth_config import" = "from backend.core.security import"
        }
        $modified = $false
        foreach ($old in $patterns.Keys) {
            if ($content -match [regex]::Escape($old)) {
                $content = $content -replace [regex]::Escape($old), $patterns[$old]
                $modified = $true
                $fixes += "Fixed: $($file.FullName) - $old -> $($patterns[$old])"
            }
        }
        if ($modified) { 
            # Sauvegarde avec un backup simple
            Copy-Item $file.FullName "$($file.FullName).bak" -Force
            Set-Content -Path $file.FullName -Value $content -Encoding UTF8 
        }
    }
}

# --------------------------
# Step 5: Clean up & Final report
# --------------------------
# Nettoyage du fichier temporaire
Remove-Item $tempScriptPath -ErrorAction SilentlyContinue

Write-Host "`n--- Rapport Final ---" -ForegroundColor $Cyan
Write-Host "Erreurs trouvées: $($errors.Count)" -ForegroundColor $(if ($errors.Count -gt 0) { $Red } else { $Green })

if ($errors.Count -gt 0) {
    Write-Host "`nListe des erreurs:" -ForegroundColor $Red
    $errors | ForEach-Object { Write-Host "   • $_" -ForegroundColor $Red }
}

if ($FixImports -and $fixes.Count -gt 0) {
    Write-Host "`n✅ Corrections appliquées (avec -FixImports):" -ForegroundColor $Green
    $fixes | ForEach-Object { Write-Host "   • $_" -ForegroundColor $Green }
}

