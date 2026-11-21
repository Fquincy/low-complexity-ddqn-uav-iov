$ErrorActionPreference = 'Stop'

$root = 'E:\Fitz Thesis docs\Low Complexity'
$runRoot = Join-Path $root 'development_runs\assignment_price_development_2026-11-16_01'
$liveStatus = Join-Path $runRoot 'LIVE_STATUS.txt'
$monitorJson = Join-Path $runRoot 'AUDIT_MONITOR.json'
$activePath = Join-Path $root 'validation\assignment_price_evaluation_continuation_v2_readiness_2026-11-17_01\ACTIVE_AUDIT_ARCHIVE.json'
$stderrPath = Join-Path $root 'validation\assignment_price_evaluation_continuation_v2_readiness_2026-11-17_01\audit_archive_stderr.log'
$stdoutPath = Join-Path $root 'validation\assignment_price_evaluation_continuation_v2_readiness_2026-11-17_01\audit_archive_stdout.log'
$completionPath = Join-Path $runRoot 'audited_completion\completion.json'
$auditReportPath = Join-Path $runRoot 'audited_completion\saved_artifact_audit.json'
$archivePath = 'D:\Fitz Thesis Research Archive\assignment_price_development_2026-11-16_01'
$archiveVerification = Join-Path $archivePath 'ARCHIVE_VERIFICATION.json'

function File-Length([string]$path) {
    if (Test-Path -LiteralPath $path) { return (Get-Item -LiteralPath $path).Length }
    return 0
}

do {
    $active = Get-Content -LiteralPath $activePath -Raw | ConvertFrom-Json
    $worker = Get-Process -Id $active.worker_pid -ErrorAction SilentlyContinue
    $launcher = Get-Process -Id $active.launcher_pid -ErrorAction SilentlyContinue
    $completionExists = Test-Path -LiteralPath $completionPath
    $auditReportExists = Test-Path -LiteralPath $auditReportPath
    $archiveDirectoryExists = Test-Path -LiteralPath $archivePath
    $archiveVerified = Test-Path -LiteralPath $archiveVerification
    $stderrBytes = File-Length $stderrPath
    $stdoutBytes = File-Length $stdoutPath

    if ($archiveVerified) {
        $stage = 'AUDITED_AND_ARCHIVED_COMPLETE'
    }
    elseif ($completionExists -and $worker) {
        $stage = 'ARCHIVE_RUNNING'
    }
    elseif ($worker) {
        $stage = 'FULL_ARTIFACT_AUDIT_RUNNING'
    }
    elseif ($stderrBytes -gt 0) {
        $stage = 'AUDIT_OR_ARCHIVE_STOPPED_WITH_ERROR'
    }
    elseif ($completionExists) {
        $stage = 'AUDIT_COMPLETE_ARCHIVE_NOT_VERIFIED'
    }
    else {
        $stage = 'AUDIT_PROCESS_STOPPED_NEEDS_REVIEW'
    }

    $now = [DateTimeOffset]::Now
    $started = [DateTimeOffset]::Parse([string]$active.started_at)
    $drives = Get-PSDrive -Name D,E | Select-Object Name,Free,Used
    $snapshot = [ordered]@{
        updated_at = $now.ToString('o')
        stage = $stage
        launcher_running = $null -ne $launcher
        worker_running = $null -ne $worker
        launcher_pid = [int]$active.launcher_pid
        worker_pid = [int]$active.worker_pid
        elapsed_hours = [math]::Round(($now - $started).TotalHours, 2)
        worker_cpu_seconds = if ($worker) { [math]::Round($worker.CPU, 1) } else { $null }
        worker_working_set_mb = if ($worker) { [math]::Round($worker.WorkingSet64 / 1MB, 1) } else { $null }
        stderr_bytes = $stderrBytes
        stdout_bytes = $stdoutBytes
        run_manifest_present = Test-Path -LiteralPath (Join-Path $runRoot 'run\manifest.json')
        audit_report_present = $auditReportExists
        audited_completion_present = $completionExists
        archive_directory_present = $archiveDirectoryExists
        archive_verification_present = $archiveVerified
        training_slots_completed = 600000
        evaluation_slots_completed = 151680
        optimizer_updates_completed = 523275
        final_checkpoints = 75
        audit_percent = $null
        audit_percent_note = 'The auditor has no checkpoint counter, so a defensible completion percentage is unavailable.'
        drive_space = $drives
    }
    $snapshot | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $monitorJson -Encoding UTF8

    $d = $drives | Where-Object Name -eq 'D'
    $e = $drives | Where-Object Name -eq 'E'
    @(
        "STAGE: $stage"
        "Updated: $($now.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
        "Audit worker: PID $($active.worker_pid); running=$($null -ne $worker)"
        "Elapsed wall time: $($snapshot.elapsed_hours) hours"
        "Worker CPU time: $($snapshot.worker_cpu_seconds) seconds"
        "Worker memory: $($snapshot.worker_working_set_mb) MB"
        "Errors logged: $stderrBytes bytes"
        "Run manifest present: $($snapshot.run_manifest_present)"
        "Audit report present: $auditReportExists"
        "Audited completion present: $completionExists"
        "Archive directory present: $archiveDirectoryExists"
        "Archive SHA256 verification present: $archiveVerified"
        "E: free: $([math]::Round($e.Free / 1GB, 2)) GiB"
        "D: free: $([math]::Round($d.Free / 1GB, 2)) GiB"
        'Training: 600,000 / 600,000 slots; evaluation: 151,680 / 151,680 slots.'
        'Audit percentage: unavailable because this auditor does not emit internal checkpoint counts.'
        "Detailed machine-readable monitor: $monitorJson"
        'No retraining, reevaluation or retiming is occurring.'
    ) | Set-Content -LiteralPath $liveStatus -Encoding UTF8

    if ($stage -notin @('FULL_ARTIFACT_AUDIT_RUNNING','ARCHIVE_RUNNING')) { break }
    Start-Sleep -Seconds 60
} while ($true)
