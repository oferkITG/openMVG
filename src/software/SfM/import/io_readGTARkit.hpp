// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2018 Yan Qingsong, Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef IO_READ_GT_ARKIT_HPP
#define IO_READ_GT_ARKIT_HPP

#include "io_readGTInterface.hpp"
#include "io_loadImages.hpp"

#include "openMVG/cameras/PinholeCamera.hpp"
#include <string>
//#include <fstream>
//#include <iomanip>
#include <math.h>

//timestamp, frame id, flx, fly, px, py
struct Cameras_Data_ARkit
{
    double timestamp_;
    int id_;
    std::string model_name_;
    int width_;
    int height_;
    std::vector<double> parameter_;
};


// ARkit's Data store all the data in two file under the gt_dir
// ARPoses.txt    -> camera location (timestamp, x, y, z, qw, qx, qy, qz)
// Frames.txx     -> frames timestamp, frame id, flx, fly, px, py
class SfM_Data_GT_Loader_ARkit : public SfM_Data_GT_Loader_Interface
{
private:
    std::vector<cameras::PinholeCamera> cameras_data_; // Store all the camera information
    std::map<double, Cameras_Data_ARkit> camera_datas; // Store all the dataset camera data
public:
    bool loadGT() override
    {
        // Check all the files under the path
        const std::vector<std::string> gt_files = stlplus::folder_files(this->gt_dir_);

        // Make sure there we have the desired file on disk
        if (!std::count(gt_files.cbegin(), gt_files.cend(), std::string("Frames.txt"))
            || !std::count(gt_files.cbegin(), gt_files.cend(), std::string("ARposes.txt")))
        {
            std::cerr << "Error: Maybe give wrong gt_dir!" << std::endl
                << "Make sure there in only two files(ARposes.txt Frames.txt) under the gt_dir!" << std::endl;
            return false;
        }

        // Read the camera file
        // Fix name "Frames.txt"
        std::ifstream camera_data_file(stlplus::create_filespec(this->gt_dir_, "Frames.txt"), std::ifstream::in);
        if (!camera_data_file)
        {
            std::cerr << "Error: Failed to open file '" << stlplus::create_filespec(this->gt_dir_, "Frames.txt") << "' for reading" << std::endl;
            return false;
        }
        while (camera_data_file)
        {
            std::string line;
            std::getline(camera_data_file, line);
            if (line.size() == 0 || line[0] == '#')
            {
                continue;
            }

            Cameras_Data_ARkit temp_camera;
            temp_camera.width_ = 1920;
            temp_camera.height_ = 1440;
            temp_camera.model_name_ = "PINHOLE";

            std::string substring;

            std::istringstream line_stream(line);
            std::getline(line_stream, substring, ',');
            temp_camera.timestamp_ = stod(substring);
            std::getline(line_stream, substring, ',');
            temp_camera.id_ = stoi(substring);
            
            while (std::getline(line_stream, substring, ',')) {

                temp_camera.parameter_.push_back(stod(substring));
            }

            camera_datas.insert({ temp_camera.timestamp_,temp_camera });
        }
        camera_data_file.close();

        // Load the gt_data from the file
        std::ifstream gt_file(stlplus::create_filespec(this->gt_dir_, "ARposes.txt"), std::ifstream::in);
        if (!gt_file)
        {
            return false;
        }
        int image_number_count = 0;
        while (gt_file)
        {
            std::string line;
            std::getline(gt_file, line);
            if (line.empty() || line[0] == '#')
            {
                continue;
            }
            image_number_count++;
            std::getline(gt_file, line);
        }
        cameras_data_.reserve(image_number_count);

        gt_file.clear(std::ios::goodbit);
        gt_file.seekg(std::ios::beg);

        Vec3 prev_t;
        Eigen::Quaterniond prev_qt;

        while (gt_file)
        {
            std::string line;
            std::getline(gt_file, line);
            if (line.empty() || line[0] == '#')
            {
                continue;
            }

            // Read image info line.
            Mat3 K, R;
            Vec3 t;
            Eigen::Quaterniond quaternionf_rotation;
            std::istringstream image_stream(line);

            double timestamp;
            std::string substring;
            std::getline(image_stream, substring, ',');
            timestamp = stod(substring);
            std::getline(image_stream, substring, ',');
            t(0, 0) = stod(substring);
            std::getline(image_stream, substring, ',');
            t(1, 0) = stod(substring);
            std::getline(image_stream, substring, ',');
            t(2, 0) = stod(substring);
            
            std::getline(image_stream, substring, ',');
            quaternionf_rotation.w() = stod(substring);
            std::getline(image_stream, substring, ',');
            quaternionf_rotation.x() = stod(substring);
            std::getline(image_stream, substring, ',');
            quaternionf_rotation.y() = stod(substring);
            std::getline(image_stream, substring, ',');
            quaternionf_rotation.z() = stod(substring);

            //t.z() = -t.z();
            //t.y() = -t.y();
            
            R = quaternionf_rotation.toRotationMatrix();
            const Mat3 transform_matrix = (Mat3() << 1, 1, 1, -1, -1, -1, -1, -1, -1).finished();
            Mat3 Rtranspose = R.transpose();
            R = transform_matrix.cwiseProduct(Rtranspose);
            
            t = -R * t;

            if (camera_datas.find(timestamp) == camera_datas.end())
                continue;

            double del_qt = quaternionf_rotation.angularDistance(prev_qt); //radian
            double del_t = sqrt((t.x() - prev_t.x()) * (t.x() - prev_t.x()) + (t.y() - prev_t.y()) * (t.y() - prev_t.y()) + (t.z() - prev_t.z()) * (t.z() - prev_t.z()));

            //if (del_qt > 10.0 / 180.0 * M_PI || del_t > 1) {
                prev_qt = quaternionf_rotation;
                prev_t = t;

                Cameras_Data_ARkit temp_camera = camera_datas[timestamp];
                double focus = (temp_camera.parameter_[0] + temp_camera.parameter_[1]) / 2;
                K << focus, 0, temp_camera.parameter_[2],
                    0, focus, temp_camera.parameter_[3],
                    0, 0, 1;

                // Read feature observations line.
                // No use for us
                std::getline(gt_file, line);

                Mat34 P;
                P_From_KRt(K, R, t, &P);
                cameras_data_.emplace_back(P);

                int image_id = temp_camera.id_;
                std::string image_name = "frame" + std::to_string(image_id) + ".jpg";

                // Parse image name
                images_.emplace_back(stlplus::filename_part(image_name));
            //}
        }
        gt_file.close();

        return true;
    }

    bool loadImages() override
    {
        return LoadImages(this->image_dir_, this->images_, this->cameras_data_, this->sfm_data_);
    }
};


#endif // IO_READ_GT_ARKIT_HPP
