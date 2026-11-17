@echo off
setlocal
set "WOUAF_PRODUCT_ROOT=%~dp0"
if "%~1"=="" (echo Usage: predict_windows.cmd IMAGE [v3^|v1^|v2] & exit /b 2)
set "WOUAF_PRODUCT_IMAGE=%~f1"
set "WOUAF_PRODUCT_VERSION=%~2"
powershell.exe -NoLogo -NoProfile -Command "$ErrorActionPreference='Stop'; $root=$env:WOUAF_PRODUCT_ROOT; $file=Join-Path $root 'outputs/portable_install.json'; if (!(Test-Path -LiteralPath $file)) { throw 'Run setup_windows.cmd first' }; $receipt=Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json; $version=$env:WOUAF_PRODUCT_VERSION; if (!$version) { $version='v3' }; if ($version -notin @('v1','v2','v3')) { throw 'Version must be v1, v2 or v3' }; $output=Join-Path $root ('outputs/prediction_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'.json'); & $receipt.python -I -X utf8 (Join-Path $root 'wouaf.py') predict --image $env:WOUAF_PRODUCT_IMAGE --version $version --output $output; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; Write-Host ('SAVED_PREDICTION '+$output);"
exit /b %errorlevel%
