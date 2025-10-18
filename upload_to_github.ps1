# DeepCSAT GitHub Upload Script for PowerShell
# This script helps upload the DeepCSAT project to GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DeepCSAT GitHub Upload Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "✅ Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git is not installed. Please install Git from https://git-scm.com" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in a git repository
if (Test-Path ".git") {
    Write-Host "✅ Git repository already initialized" -ForegroundColor Green
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
}

# Add all files
Write-Host "Adding files to Git..." -ForegroundColor Yellow
git add .

# Check git status
Write-Host "Git status:" -ForegroundColor Yellow
git status

# Create initial commit
Write-Host "Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: DeepCSAT project setup"
Write-Host "✅ Initial commit created" -ForegroundColor Green

# Get repository URL from user
Write-Host ""
Write-Host "Please create a new repository on GitHub.com first:" -ForegroundColor Cyan
Write-Host "1. Go to https://github.com" -ForegroundColor White
Write-Host "2. Click 'New repository'" -ForegroundColor White
Write-Host "3. Name it 'DeepCSAT'" -ForegroundColor White
Write-Host "4. Don't initialize with README, .gitignore, or license" -ForegroundColor White
Write-Host "5. Click 'Create repository'" -ForegroundColor White
Write-Host ""

$repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/DeepCSAT.git)"

if ($repoUrl -eq "") {
    Write-Host "❌ No repository URL provided. Exiting." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Add remote origin
Write-Host "Adding remote origin..." -ForegroundColor Yellow
git remote add origin $repoUrl
Write-Host "✅ Remote origin added" -ForegroundColor Green

# Rename branch to main
Write-Host "Renaming branch to main..." -ForegroundColor Yellow
git branch -M main
Write-Host "✅ Branch renamed to main" -ForegroundColor Green

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 SUCCESS! Your DeepCSAT project has been uploaded to GitHub!" -ForegroundColor Green
    Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Visit your repository on GitHub" -ForegroundColor White
    Write-Host "2. Add a description and topics" -ForegroundColor White
    Write-Host "3. Enable GitHub Pages if desired" -ForegroundColor White
    Write-Host "4. Share your repository!" -ForegroundColor White
} else {
    Write-Host "❌ Failed to push to GitHub. Please check your credentials and try again." -ForegroundColor Red
    Write-Host "You may need to set up authentication:" -ForegroundColor Yellow
    Write-Host "1. Use GitHub CLI: gh auth login" -ForegroundColor White
    Write-Host "2. Or use personal access token" -ForegroundColor White
    Write-Host "3. Or use SSH keys" -ForegroundColor White
}

Write-Host ""
Read-Host "Press Enter to exit"
