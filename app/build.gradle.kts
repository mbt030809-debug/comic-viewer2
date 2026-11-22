plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.example.comicviewer"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.example.comicviewer"
        minSdk = 24
        targetSdk = 34
        versionCode = 126
        versionName = "2.01"
        ndk { abiFilters += listOf("arm64-v8a") }   // wm2 시험 빌드: 64비트 전용(APK 용량 축소). main 에 합칠 때는 armeabi-v7a 복원 여부를 확인
    }
    signingConfigs {
        getByName("debug") {
            storeFile = file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }
    buildTypes {
        // wm2 시험 빌드: 패키지 이름을 달리해 기존 앱과 나란히 설치(서명·버전 충돌 방지). main 에 합칠 때는 제거
        getByName("debug") { applicationIdSuffix = ".wm2"; versionNameSuffix = "-wm2" }
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.3")
}
