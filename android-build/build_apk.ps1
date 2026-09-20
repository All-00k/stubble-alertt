$ErrorActionPreference = "Stop"

$BUILD_TOOLS = "D:\Android\build-tools\35.0.0"
$PLATFORM_JAR = "D:\Android\platforms\android-36\android.jar"
$KEYTOOL = "C:\Program Files\Java\jdk-26.0.1\bin\keytool.exe"
$BASE = "D:\Stubble-Alert\android-build"
$OUTPUT_APK = "D:\Stubble-Alert\Stubble-Alert.apk"

Write-Host "=== STEP 1: Generate R.java ==="
& "$BUILD_TOOLS\aapt.exe" package -f -m `
    -J "$BASE\gen" `
    -M "$BASE\AndroidManifest.xml" `
    -S "$BASE\res" `
    -I "$PLATFORM_JAR"

Write-Host "=== STEP 2: Compile Java Classes ==="
$classesDir = "$BASE\build\classes"
if (-not (Test-Path $classesDir)) { New-Item -ItemType Directory -Path $classesDir -Force }

javac --release 8 -cp "$PLATFORM_JAR;$BASE\gen" -d $classesDir `
    "$BASE\gen\com\stubblealert\app\R.java" `
    "$BASE\src\com\stubblealert\app\MainActivity.java"

Write-Host "=== STEP 3: Convert to classes.dex (d8) ==="
$classFiles = Get-ChildItem "$classesDir\com\stubblealert\app\*.class" | ForEach-Object { $_.FullName }
& "$BUILD_TOOLS\d8.bat" --lib "$PLATFORM_JAR" --output "$BASE\build" $classFiles

if (-not (Test-Path "$BASE\build\classes.dex")) {
    throw "classes.dex not generated!"
}
Write-Host "classes.dex successfully created."

Write-Host "=== STEP 4: Package Resources & Assets with AAPT ==="
$unalignedApk = "$BASE\build\unaligned.apk"
if (Test-Path $unalignedApk) { Remove-Item $unalignedApk -Force }

& "$BUILD_TOOLS\aapt.exe" package -f `
    -M "$BASE\AndroidManifest.xml" `
    -S "$BASE\res" `
    -A "$BASE\assets" `
    -I "$PLATFORM_JAR" `
    -F $unalignedApk

Write-Host "=== STEP 5: Add classes.dex into APK ==="
Push-Location "$BASE\build"
& "$BUILD_TOOLS\aapt.exe" add unaligned.apk classes.dex
Pop-Location

Write-Host "=== STEP 6: ZipAlign APK ==="
if (Test-Path $OUTPUT_APK) { Remove-Item $OUTPUT_APK -Force }
& "$BUILD_TOOLS\zipalign.exe" -f -p 4 $unalignedApk $OUTPUT_APK

Write-Host "=== STEP 7: Generate Debug Keystore if missing ==="
$keystore = "$BASE\debug.keystore"
if (-not (Test-Path $keystore)) {
    & $KEYTOOL -genkey -v -keystore $keystore `
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 `
        -storepass android -keypass android `
        -dname "CN=StubbleAlert, OU=Engineering, O=StubbleAlert, L=Delhi, S=Delhi, C=IN"
}

Write-Host "=== STEP 8: Sign APK with apksigner ==="
& "$BUILD_TOOLS\apksigner.bat" sign `
    --ks $keystore `
    --ks-pass pass:android `
    --key-pass pass:android `
    $OUTPUT_APK

Write-Host "=== STEP 9: Verify APK ==="
& "$BUILD_TOOLS\apksigner.bat" verify -v $OUTPUT_APK

$apkItem = Get-Item $OUTPUT_APK
Write-Host "`nSUCCESS! APK generated and signed:"
Write-Host "Path: $($apkItem.FullName)"
Write-Host "Size: $([math]::Round($apkItem.Length / 1MB, 2)) MB ($($apkItem.Length) bytes)"
