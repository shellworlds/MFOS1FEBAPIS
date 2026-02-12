Write-Host "Installing MFOS1FEBAPIS Quantum Framework via WSL2..." -ForegroundColor Green
$wsl = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
if ($wsl.State -ne "Enabled") {
Write-Host "Enabling WSL..." -ForegroundColor Yellow
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All
wsl --set-default-version 2
}
wsl -d Ubuntu -e bash -c "curl -sSL https://raw.githubusercontent.com/shellworlds/MFOS1FEBAPIS/main/client_installers/universal/install.sh | bash"

Write-Host "Installation initiated. Follow prompts in WSL terminal." -ForegroundColor Green
