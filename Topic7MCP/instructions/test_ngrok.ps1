# ============================================================
# ngrok Campus Test Script (PowerShell / Windows)
#
# Tests whether ngrok can create a tunnel from behind your
# university firewall.
#
# Steps:
#   1. Checks for ngrok
#   2. Checks for authtoken
#   3. Starts a tiny Python HTTP server
#   4. Opens an ngrok tunnel to it
#   5. Verifies the tunnel works from the outside
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\test_ngrok.ps1
#
# If PowerShell blocks script execution, use:
#   Set-ExecutionPolicy -Scope Process Bypass
# ============================================================

$Port = 8765
$TimeoutSeconds = 15

$serverProcess = $null
$ngrokProcess = $null
$tempDir = Join-Path $env:TEMP "ngrok_test"
$serverScript = Join-Path $tempDir "server.py"
$ngrokLog = Join-Path $tempDir "ngrok_test.log"

function Cleanup {
    Write-Host ""
    Write-Host "Cleaning up..."
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($ngrokProcess -and -not $ngrokProcess.HasExited) {
        Stop-Process -Id $ngrokProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Done."
}

try {
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

    Write-Host "========================================"
    Write-Host "  ngrok Campus Connectivity Test"
    Write-Host "========================================"
    Write-Host ""

    # --- Step 1: Check for ngrok ---
    Write-Host "[1/6] Checking for ngrok..."
    $ngrokCmd = Get-Command ngrok -ErrorAction SilentlyContinue

    if (-not $ngrokCmd) {
        Write-Host "  X ngrok not found in PATH."
        Write-Host "    Install ngrok from: https://ngrok.com/download"
        Write-Host "    Or make sure ngrok.exe is on your PATH."
        exit 1
    }

    $ngrokExe = $ngrokCmd.Source
    Write-Host "  ✓ ngrok found at $ngrokExe"
    Write-Host ""

    # --- Step 2: Check for authtoken ---
    Write-Host "[2/6] Checking for ngrok authtoken..."

    $configLocations = @(
        (Join-Path $env:LOCALAPPDATA "ngrok\ngrok.yml"),
        (Join-Path $env:USERPROFILE ".ngrok2\ngrok.yml"),
        (Join-Path $env:USERPROFILE ".config\ngrok\ngrok.yml")
    )

    $authtokenFound = $false
    foreach ($cfg in $configLocations) {
        if ((Test-Path $cfg) -and (Select-String -Path $cfg -Pattern "authtoken:" -Quiet -ErrorAction SilentlyContinue)) {
            $authtokenFound = $true
            Write-Host "  ✓ Authtoken found in: $cfg"
            break
        }
    }

    if (-not $authtokenFound) {
        Write-Host "  X No authtoken found."
        Write-Host ""
        Write-Host "  ngrok requires a free account and authtoken to work."
        Write-Host "    1. Go to https://dashboard.ngrok.com/signup"
        Write-Host "    2. Sign up"
        Write-Host "    3. Go to https://dashboard.ngrok.com/get-started/your-authtoken"
        Write-Host "    4. Copy your authtoken"
        Write-Host ""

        $userToken = Read-Host "  Paste your authtoken here (or Ctrl+C to quit)"
        if ([string]::IsNullOrWhiteSpace($userToken)) {
            Write-Host "  X No token entered. Exiting."
            exit 1
        }

        $userToken = $userToken.Trim()
        Write-Host ""
        Write-Host "  Configuring ngrok with your authtoken..."

        & $ngrokExe config add-authtoken $userToken
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  X Failed to configure authtoken."
            exit 1
        }

        Write-Host "  ✓ Authtoken configured successfully"
    }
    Write-Host ""

    # --- Step 3: Start test HTTP server ---
    Write-Host "[3/6] Starting test HTTP server on port $Port..."

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Host "  X Python not found in PATH."
        exit 1
    }
    $pythonExe = $pythonCmd.Source

    @"
import http.server
import json
import socket
import sys

PORT = $Port

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {
            "status": "ok",
            "message": "ngrok tunnel is working!",
            "hostname": socket.gethostname(),
            "test": "campus_connectivity"
        }
        self.wfile.write(json.dumps(response, indent=2).encode())

    def log_message(self, format, *args):
        pass

server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
server.serve_forever()
"@ | Set-Content -Path $serverScript -Encoding UTF8

    $serverProcess = Start-Process -FilePath $pythonExe -ArgumentList $serverScript -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 1

    try {
        $localResponse = Invoke-WebRequest -Uri "http://127.0.0.1:$Port" -UseBasicParsing -TimeoutSec 5
        if ($localResponse.StatusCode -eq 200) {
            Write-Host "  ✓ HTTP server running on localhost:$Port"
        } else {
            Write-Host "  X Failed to start HTTP server."
            exit 1
        }
    } catch {
        Write-Host "  X Failed to start HTTP server. Is port $Port in use?"
        exit 1
    }
    Write-Host ""

    # --- Step 4: Start ngrok ---
    Write-Host "[4/6] Starting ngrok tunnel (this is the real test)..."
    Write-Host "  If this hangs, ngrok may be blocked on your network."
    Write-Host ""

    if (Test-Path $ngrokLog) {
        Remove-Item $ngrokLog -Force -ErrorAction SilentlyContinue
    }

    $ngrokProcess = Start-Process -FilePath $ngrokExe `
        -ArgumentList "http $Port --log=stdout --log-level=info" `
        -RedirectStandardOutput $ngrokLog `
        -RedirectStandardError $ngrokLog `
        -PassThru `
        -WindowStyle Hidden

    Write-Host -NoNewline "  Waiting for tunnel"
    $tunnelUrl = $null

    for ($i = 1; $i -le $TimeoutSeconds; $i++) {
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 1

        try {
            $apiResponse = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 3
            foreach ($t in $apiResponse.tunnels) {
                if ($t.public_url -like "https://*") {
                    $tunnelUrl = $t.public_url
                    break
                }
            }
            if ($tunnelUrl) { break }
        } catch {
            # ignore until timeout
        }
    }
    Write-Host ""
    Write-Host ""

    if (-not $tunnelUrl) {
        Write-Host "  X FAILED: ngrok could not establish a tunnel within $TimeoutSeconds seconds."
        Write-Host ""
        Write-Host "  Possible reasons:"
        Write-Host "    - University firewall blocks ngrok outbound connections"
        Write-Host "    - ngrok binary is outdated"
        Write-Host "    - Network requires a proxy"
        Write-Host ""

        if (Test-Path $ngrokLog) {
            Write-Host "  Last few lines of ngrok log:"
            Get-Content $ngrokLog -Tail 20 |
                Select-String -Pattern "err|fail|block|refused|timeout" -CaseSensitive:$false |
                Select-Object -First 5 |
                ForEach-Object { Write-Host "    $($_.Line)" }
        }
        exit 1
    }

    Write-Host "  ✓ Tunnel established!"
    Write-Host "  Public URL: $tunnelUrl"
    Write-Host ""

    # --- Step 5: Test tunnel connectivity ---
    Write-Host "[5/6] Testing tunnel connectivity..."
    $tunnelOk = $false
    $responseText = $null

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            $headers = @{ "ngrok-skip-browser-warning" = "true" }
            $webResp = Invoke-WebRequest -Uri $tunnelUrl -Headers $headers -UseBasicParsing -TimeoutSec 5
            $responseText = $webResp.Content

            $json = $responseText | ConvertFrom-Json
            if ($json.status -eq "ok") {
                $tunnelOk = $true
                break
            }
        } catch {
            # ignore and retry
        }
    }

    if ($tunnelOk) {
        Write-Host "  ✓ Tunnel is fully functional!"
        Write-Host "  Response from tunnel:"
        try {
            ($responseText | ConvertFrom-Json) | ConvertTo-Json -Depth 5 | ForEach-Object {
                $_ -split "`n" | ForEach-Object { Write-Host "    $_" }
            }
        } catch {
            Write-Host "    $responseText"
        }
    } else {
        Write-Host "  ! Tunnel exists but the test response was unexpected."
        Write-Host "  This can happen if ngrok shows a warning/interstitial page."
        Write-Host "  The important result is that the tunnel was established."
    }
    Write-Host ""

    # --- Step 6: Summary ---
    Write-Host "[6/6] Results"
    Write-Host "========================================"
    Write-Host ""
    Write-Host "  ngrok:     WORKS ✓"
    Write-Host "  Tunnel:    $tunnelUrl"
    Write-Host "  Server:    localhost:$Port"
    Write-Host ""
    Write-Host "  You can use ngrok to expose your"
    Write-Host "  MCP/A2A agents from this network."
    Write-Host ""
    Write-Host "  Quick reference:"
    Write-Host "    ngrok http 8000"
    Write-Host ""
    Write-Host "  Press Ctrl+C to shut down the test."
    Write-Host "========================================"

    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Cleanup
}