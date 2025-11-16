param(
    [Parameter(Mandatory=$false)]
    [Switch]$DryRun,

    [Parameter(Mandatory=$false)]
    [Switch]$Force
)

$BaseDir = "LEXIBOT_PRO/backend"

# --- Fichiers à Remplir (Squelettes) ---

$FilesToFill = @(
    @{
        Path = "$BaseDir/api/routers/calendar.py";
        Content = "from fastapi import APIRouter, Depends; router = APIRouter(tags=['📅 Calendrier']); @router.get('/')\nasync def get_calendar_info():\n    return {'status': 'Calendar router OK'}"
    },
    @{
        Path = "$BaseDir/api/routers/auth.py";
        Content = "from fastapi import APIRouter; from backend.models.models import UserRead, UserCreate; router = APIRouter(tags=['🔐 Authentification']);\n# NOTE: La logique FastAPI-Users sera incluse ici (ex: auth_backend, register_router)\n@router.get('/')\nasync def get_auth_info():\n    return {'status': 'Auth router OK'}"
    },
    @{
        Path = "$BaseDir/schemas/auth.py";
        Content = "from pydantic import BaseModel\n# NOTE: Les schémas Pydantic pour l'auth vont ici (ex: UserCreate, UserRead, etc.)\nclass AuthInfo(BaseModel):\n    message: str = 'Auth schemas OK'"
    }
)

function Run-Filling {
    Write-Host "--- Démarrage du Remplissage des Fichiers Vides ---" -ForegroundColor Yellow
    $FilledCount = 0

    foreach ($Item in $FilesToFill) {
        $Path = $Item.Path
        $Content = $Item.Content

        if (Test-Path $Path) {
            $FileContent = Get-Content $Path
            # Remplir uniquement si le fichier est vide (ou forcé)
            if ($Force -or -not $FileContent) {
                if (-not $DryRun) {
                    $Content | Set-Content -Path $Path -Force
                }
                Write-Host "✅ Fichier rempli : $Path" -ForegroundColor Green
                $FilledCount++
            } else {
                Write-Host "   Fichier non vide : $Path (ignoré)" -ForegroundColor DarkYellow
            }
        } else {
            Write-Host "   Fichier manquant : $Path (Veuillez lancer reorganize_lexibot.ps1 d'abord)" -ForegroundColor Red
        }
    }

    Write-Host "--- Remplissage TERMINÉ ---" -ForegroundColor Green
    Write-Host "Total de fichiers remplis : $FilledCount" -ForegroundColor Yellow
}

# --- Logique d'exécution ---

if ($DryRun) {
    Write-Host "Mode simulation de remplissage activé. Aucune écriture de fichier." -ForegroundColor Cyan
}

Run-Filling
