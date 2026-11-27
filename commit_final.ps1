<#
.SYNOPSIS
    Stages and commits all pending Rakshak repository changes in logical, action-based commits.
    Explicitly excludes this script (commit_final.ps1) from staging and committing.

.EXAMPLE
    .\commit_final.ps1 -DryRun
    .\commit_final.ps1
#>
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    throw "Expected a .git repository at $RepoRoot"
}

$Commits = @(
    @{
        Subject = "feat(ingest,translit): add homoglyph folding, Verhoeff check, and path hygiene"
        Body    = "Normalize zero-width/homoglyph obfuscation, add Verhoeff Aadhaar checksum helper, and harden filename/CSV ingestion."
        Files   = @("rakshak/translit.py", "rakshak/ingest.py")
    },
    @{
        Subject = "feat(schema,clues): add compositional column profiling and date-order pooling"
        Body    = "Infer unseen column/key roles via token semantics + value stats, pool DMY/MDY/ISO orders, and scan bare Verhoeff Aadhaars."
        Files   = @("rakshak/contracts.py", "rakshak/schema_profiler.py", "rakshak/clue_extractor.py")
    },
    @{
        Subject = "fix(linkage): enforce all-name cluster guard and age +/-2 corroboration"
        Body    = "Switch _clusters_compatible from any() to all() on multi-token names and enforce +/-2 yr age match in Corroborator 4."
        Files   = @("rakshak/linkage.py")
    },
    @{
        Subject = "feat(privacy): add post-widening joint k/l checks and YYYY DOB truncation"
        Body    = "Verify joint k>=5 and l>=2 condition diversity after widening, drop singleton rare gender, and truncate standard DOB to YYYY."
        Files   = @("rakshak/planner.py", "rakshak/privacy.py")
    },
    @{
        Subject = "feat(ner): add 4-tag offline CRF fallback with Hinglish and case training"
        Body    = "Add PER/REL/LOC/ID CRF decoder with soft chemical features, retrain on Hinglish/lowercase/initial OOD patterns, and declare pycrfsuite."
        Files   = @("requirements.txt", "rakshak/ner_crf.py", "scripts/train_ner_crf.py", "models/pii_ner.crfsuite")
    },
    @{
        Subject = "feat(detector,formats): add inline encoding scrub, Verhoeff Aadhaar, and FHIR hygiene"
        Body    = "Decode inline base64/hex/\uXXXX/HTML/%URL payloads, apply YYYY DOB & Verhoeff rules, and scrub FHIR/JSON/path leaks."
        Files   = @("main.py", "rakshak/detector.py", "rakshak/formats.py")
    },
    @{
        Subject = "test(suite): expand chaos, evasion, linkage, and joint k/l privacy tests"
        Body    = "Verify 54 tests covering unseen opaque schemas, inline evasion, all() linkage guard, YYYY DOB, and post-widening k/l invariants."
        Files   = @(
            "tests/test_phase1_translit.py",
            "tests/test_phase2_extractor.py",
            "tests/test_phase3_linkage.py",
            "tests/test_phase4_privacy.py",
            "tests/test_chaos.py"
        )
    },
    @{
        Subject = "docs(brief,readme): update README, architecture SVG, and 2-page project brief PDF"
        Body    = "Add docs/rakshak-architecture.svg, refresh README benchmarks, and update docs/brief.pdf."
        Files   = @("README.md", "docs/rakshak-architecture.svg", "docs/brief.pdf")
    }
)

if (-not $DryRun) {
    # Reset any pre-staged files so commit_final.ps1 is never accidentally committed
    git -C $RepoRoot reset --mixed --quiet
}

$CommittedCount = 0
$CoveredFiles = [System.Collections.Generic.HashSet[string]]::new()
foreach ($c in $Commits) {
    $pending = @()
    foreach ($f in $c.Files) {
        if ($f -eq "commit_final.ps1") { continue }
        $null = $CoveredFiles.Add($f)
        $st = git -C $RepoRoot status --porcelain -uall -- $f
        if ($st) {
            $pending += $f
        }
    }

    if ($pending.Count -eq 0) {
        Write-Host "[SKIP] Nothing to commit for: $($c.Subject)" -ForegroundColor DarkGray
        continue
    }

    Write-Host "`n[COMMIT] $($c.Subject)" -ForegroundColor Cyan
    foreach ($pf in $pending) {
        Write-Host "   + $pf" -ForegroundColor Green
    }

    if (-not $DryRun) {
        git -C $RepoRoot add -- @pending
        git -C $RepoRoot commit -m $c.Subject -m $c.Body
        if ($LASTEXITCODE -ne 0) {
            throw "git commit failed for: $($c.Subject)"
        }
        $CommittedCount++
    }
}

Write-Host "`n--- Remaining git status (should list only commit_final.ps1 after running) ---" -ForegroundColor Yellow
git -C $RepoRoot status --short -uall

$uncovered = @()
foreach ($line in (git -C $RepoRoot status --porcelain -uall)) {
    if ($line.Length -lt 4) { continue }
    $path = $line.Substring(3).Trim().Trim('"')
    if ($path -ne "commit_final.ps1" -and -not $CoveredFiles.Contains($path)) {
        $uncovered += $path
    }
}
if ($uncovered.Count -gt 0) {
    Write-Warning "Uncovered changes not assigned to any commit: $($uncovered -join ', ')"
} else {
    Write-Host "[OK] All changed/untracked repository files (excluding commit_final.ps1) are covered cleanly." -ForegroundColor Green
}

if (-not $DryRun -and $CommittedCount -gt 0) {
    Write-Host "`n--- Recent commits ---" -ForegroundColor Yellow
    git -C $RepoRoot log --oneline -n $CommittedCount
}
