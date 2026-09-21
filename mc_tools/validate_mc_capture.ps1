[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$FramesPath,

    [Parameter(Mandatory = $true)]
    [string]$CsvPath
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $FramesPath -PathType Container)) {
    throw "FramesPath is not a directory: $FramesPath"
}
if (-not (Test-Path -LiteralPath $CsvPath -PathType Leaf)) {
    throw "CsvPath is not a file: $CsvPath"
}

$frameExtensions = @('.png', '.jpg', '.jpeg', '.webp', '.bmp')
$frames = @(Get-ChildItem -LiteralPath $FramesPath -File | Where-Object {
    $frameExtensions -contains $_.Extension.ToLowerInvariant()
} | Sort-Object Name)
$poses = @(Import-Csv -LiteralPath $CsvPath)
$errors = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if ($frames.Count -eq 0) {
    $errors.Add('No image frames were found.')
}
if ($poses.Count -eq 0) {
    $errors.Add('No camera-pose rows were found.')
}
if ($frames.Count -ne $poses.Count) {
    $errors.Add("Frame/pose count mismatch: images=$($frames.Count), poses=$($poses.Count).")
}

$maxQuaternionError = 0.0
$maxBasisLengthError = 0.0
$maxBasisDot = 0.0

for ($i = 0; $i -lt $poses.Count; $i++) {
    $p = $poses[$i]
    if ([long]$p.frame_index -ne $i) {
        $errors.Add("Non-contiguous frame_index at CSV row $($i + 2): expected $i, got $($p.frame_index).")
        break
    }

    $values = @(
        $p.x, $p.y, $p.z, $p.quat_x, $p.quat_y, $p.quat_z, $p.quat_w,
        $p.right_x, $p.right_y, $p.right_z,
        $p.up_x, $p.up_y, $p.up_z,
        $p.back_x, $p.back_y, $p.back_z, $p.fov_deg
    ) | ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    if (@($values | Where-Object { [double]::IsNaN($_) -or [double]::IsInfinity($_) }).Count -gt 0) {
        $errors.Add("NaN or Infinity at CSV row $($i + 2).")
        continue
    }

    $q = $values[3..6]
    $r = $values[7..9]
    $u = $values[10..12]
    $b = $values[13..15]
    $qNorm = [Math]::Sqrt(($q | ForEach-Object { $_ * $_ } | Measure-Object -Sum).Sum)
    $maxQuaternionError = [Math]::Max($maxQuaternionError, [Math]::Abs($qNorm - 1.0))

    foreach ($axis in @($r, $u, $b)) {
        $axisNorm = [Math]::Sqrt(($axis | ForEach-Object { $_ * $_ } | Measure-Object -Sum).Sum)
        $maxBasisLengthError = [Math]::Max($maxBasisLengthError, [Math]::Abs($axisNorm - 1.0))
    }
    $dotRU = $r[0]*$u[0] + $r[1]*$u[1] + $r[2]*$u[2]
    $dotRB = $r[0]*$b[0] + $r[1]*$b[1] + $r[2]*$b[2]
    $dotUB = $u[0]*$b[0] + $u[1]*$b[1] + $u[2]*$b[2]
    $maxBasisDot = [Math]::Max($maxBasisDot,
        [Math]::Max([Math]::Abs($dotRU), [Math]::Max([Math]::Abs($dotRB), [Math]::Abs($dotUB))))
}

if ($maxQuaternionError -gt 0.001) {
    $errors.Add(("Quaternion normalization error is too large: {0:E3}." -f $maxQuaternionError))
}
if ($maxBasisLengthError -gt 0.001 -or $maxBasisDot -gt 0.001) {
    $errors.Add(("Camera basis is not orthonormal: length_error={0:E3}, max_dot={1:E3}." -f $maxBasisLengthError, $maxBasisDot))
}

$metaPath = [IO.Path]::ChangeExtension($CsvPath, '.metadata.json')
$metadata = $null
if (Test-Path -LiteralPath $metaPath -PathType Leaf) {
    $metadata = Get-Content -LiteralPath $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($poses.Count -gt 0 -and $metadata.flashback_export) {
        $first = $poses[0]
        if ([int]$first.width -ne [int]$metadata.flashback_export.width -or
            [int]$first.height -ne [int]$metadata.flashback_export.height) {
            $errors.Add('CSV resolution does not match Flashback export metadata.')
        }
        if ($metadata.flashback_export.ssaa) {
            $warnings.Add('SSAA is enabled. Disable it for one trajectory row per output frame during Demo0.')
        }
    }
} else {
    $warnings.Add("Metadata file was not found: $metaPath")
}

$result = [ordered]@{
    valid = ($errors.Count -eq 0)
    image_frames = $frames.Count
    camera_rows = $poses.Count
    first_image = if ($frames.Count) { $frames[0].Name } else { $null }
    last_image = if ($frames.Count) { $frames[-1].Name } else { $null }
    max_quaternion_norm_error = $maxQuaternionError
    max_basis_length_error = $maxBasisLengthError
    max_basis_dot_product = $maxBasisDot
    errors = @($errors)
    warnings = @($warnings)
}

$result | ConvertTo-Json -Depth 4
if (-not $result.valid) {
    exit 1
}
