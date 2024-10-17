#include <eye_tracker/Settings.hpp>
#include <eye_tracker/frameworks/Framework.hpp>
#include <eye_tracker/frameworks/OnlineCameraFramework.hpp>
#include <eye_tracker/frameworks/VideoCameraFramework.hpp>

#include <getopt.h>
#include <string>
#include <memory>

int main(int argc, char* argv[]) {
    constexpr option options[] = {
        {"settings-path", required_argument, nullptr, 's'},
        {"user", required_argument, nullptr, 'u'},
        {"headless", no_argument, nullptr, 'h'},
        {"video", required_argument, nullptr, 'v'},
        {"calibration", required_argument, nullptr, 'c'},
        {nullptr, no_argument, nullptr, 0}
    };

    int argument{0};
    std::string settings_path{"."};
    std::string user{"default"};
    std::string shown_video_path{};
    std::string calibration_video_path{};
    bool headless{false};

    while (argument != -1) {
        argument = getopt_long(argc, argv, "s:u:v:c:h", options, nullptr);
        switch (argument) {
            case 's':
                settings_path = optarg;
                break;
            case 'u':
                user = optarg;
                break;
            case 'h':
                headless = true;
                break;
            case 'v':
                shown_video_path = optarg;
                break;
            case 'c':
                calibration_video_path = optarg;
                break;
            default:
                break;
        }
    }

    constexpr int n_cameras = 1;

    auto settings = std::make_shared<et::Settings>(settings_path);
    std::shared_ptr<et::Framework> frameworks[2];
    for (int i = 0; i < n_cameras; i++) {
        if (!et::Settings::parameters.features_params[i].contains(user)) {
            et::Settings::parameters.features_params[i][user] = et::Settings::parameters.features_params[i]["default"];
        }
        et::Settings::parameters.user_params[i] = &et::Settings::parameters.features_params[i][user];

        if (!calibration_video_path.empty()) {
            std::clog << "Calibrating from: " << calibration_video_path << std::endl;
            std::shared_ptr<et::FineTuner> fine_tuner = std::make_shared<et::FineTuner>(i);
            auto calibration_csv_path = calibration_video_path.substr(0, calibration_video_path.find_last_of('.')) + ".csv";
            fine_tuner->calculate(calibration_video_path, calibration_csv_path);
        }

        if (shown_video_path.empty()) {
            frameworks[i] = std::make_shared<et::OnlineCameraFramework>(i, headless);
        } else {
            std::clog << "Showing video from: " << shown_video_path << std::endl;
            auto shown_video_csv_path = shown_video_path.substr(0, shown_video_path.find_last_of('.')) + ".csv";
            frameworks[i] = std::make_shared<et::VideoCameraFramework>(i, headless, true, shown_video_path, shown_video_csv_path);
        }
    }

    bool finished = false;

    while (!finished) {
        const int key_pressed = cv::pollKey() & 0xFFFF;

        for (int i = 0; i < n_cameras; i++) {
            if (!frameworks[i]->analyzeNextFrame()) {
                std::clog << "Empty image. Finishing.\n";
                finished = true;
                break;
            }

            frameworks[i]->updateUi();
            switch (key_pressed) {
                case 27: // Esc
                    finished = true;
                    break;
                case 'v': {
                    frameworks[i]->startRecording();
                    break;
                }
                case 'p': {
                    frameworks[i]->captureCameraImage();
                    break;
                }
                case 'q':
                    frameworks[i]->disableImageUpdate();
                    break;
                case 'w':
                    if (!headless) {
                        frameworks[i]->switchToCameraImage();
                    }
                    break;
                case 'e':
                    if (!headless) {
                        frameworks[i]->switchToPupilThreshImage();
                    }
                    break;
                case 'r':
                    if (!headless) {
                        frameworks[i]->switchToGlintThreshImage();
                    }
                    break;
                default:
                    break;
            }

            if (frameworks[i]->shouldAppClose()) {
                finished = true;
                break;
            }
        }
    }

    cv::destroyAllWindows();
    et::Settings::saveSettings();


    return 0;
}
