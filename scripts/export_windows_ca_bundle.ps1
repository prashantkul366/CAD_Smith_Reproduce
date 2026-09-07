# Export the Windows certificate store (plus certifi) to a single PEM bundle.
#
# Why: corporate networks that inspect TLS install their root CA into the
# Windows store. Python libraries don't read that store - they use certifi -
# so requests fail with CERTIFICATE_VERIFY_FAILED. `truststore` normally
# bridges the gap, but on Python 3.12 a globally injected truststore makes
# every httpx request recurse to death instead. Handing each library an
# explicit bundle sidesteps both problems.
#
# Usage:   .\scripts\export_windows_ca_bundle.ps1
# Then it prints the three env vars to set.

$ErrorActionPreference = "Stop"
$out = Join-Path (Get-Location) "corp-ca-bundle.pem"

# Start from certifi so public CAs keep working
$certifi = python -c "import certifi; print(certifi.where())"
if (-not $?) { throw "certifi not importable - activate your venv first" }
Copy-Item $certifi $out -Force
$before = (Select-String -Path $out -Pattern "BEGIN CERTIFICATE" -AllMatches).Count

# Append every root/intermediate CA the machine trusts
$stores = @("Cert:\LocalMachine\Root", "Cert:\LocalMachine\CA",
            "Cert:\CurrentUser\Root",  "Cert:\CurrentUser\CA")
$added = 0
foreach ($store in $stores) {
    if (-not (Test-Path $store)) { continue }
    foreach ($c in Get-ChildItem $store) {
        try {
            $b64 = [Convert]::ToBase64String($c.RawData, 'InsertLineBreaks')
            Add-Content -Path $out -Encoding ascii -Value "`n# $($c.Subject)"
            Add-Content -Path $out -Encoding ascii -Value "-----BEGIN CERTIFICATE-----"
            Add-Content -Path $out -Encoding ascii -Value $b64
            Add-Content -Path $out -Encoding ascii -Value "-----END CERTIFICATE-----"
            $added++
        } catch { }
    }
}

$total = (Select-String -Path $out -Pattern "BEGIN CERTIFICATE" -AllMatches).Count
Write-Host ""
Write-Host "Wrote $out"
Write-Host "  certifi certificates : $before"
Write-Host "  Windows store added  : $added"
Write-Host "  total in bundle      : $total"
Write-Host ""
Write-Host "Set these in this PowerShell session:" -ForegroundColor Cyan
Write-Host "  `$env:SSL_CERT_FILE='$out'"
Write-Host "  `$env:AWS_CA_BUNDLE='$out'"
Write-Host "  `$env:REQUESTS_CA_BUNDLE='$out'"
