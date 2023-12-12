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

struct GPS_Data_ARkit
{
    double timestamp_;
    double latitude_;
    double longitude_;
    double horizontal_accuracy_;
    double altitude_;
    double vertical_accuracy_;
    double floor_;
    double course_;
    double speed_;
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
    std::map<double, GPS_Data_ARkit> gps_datas; // Store all the dataset gps data
    std::map<std::string, double> image_timestamps; // Store all the dataset image timestamp, 
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
            if (line.size() == 0 || line[0] == '#' || !isdigit(line[0]))
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

        // Read GPS data
        // Fix name GPS.txt
        std::ifstream gps_data_file(stlplus::create_filespec(this->gt_dir_, "GPS.txt"), std::ifstream::in);
        if (!gps_data_file)
        {
            std::cerr << "Error: Failed to open file '" << stlplus::create_filespec(this->gt_dir_, "GPS.txt") << "' for reading" << std::endl;
            return false;
        }
        int gps_count = 0;
        while (gps_data_file){
            std::string line;
            std::getline(gps_data_file, line);
            if (line.size() == 0 || line[0] == '#' || !isdigit(line[0]))
            {
                continue;
            }

            GPS_Data_ARkit temp_gps;
            std::string substring;
            std::istringstream line_stream(line);
            std::getline(line_stream, substring, ',');
            temp_gps.timestamp_ = stod(substring);
            std::getline(line_stream, substring, ',');
            temp_gps.latitude_ = stod(substring);
            std::getline(line_stream, substring, ',');
            temp_gps.altitude_ = stod(substring);
            std::getline(line_stream, substring, ',');
            temp_gps.longitude_ = stod(substring);

            gps_datas.insert({ temp_gps.timestamp_,temp_gps });

            gps_count++;
        }
        gps_data_file.close();


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
            if (line.empty() || line[0] == '#' || !isdigit(line[0]))
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
            if (line.empty() || line[0] == '#' || !isdigit(line[0]))
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

            if (del_qt > 10.0 / 180.0 * M_PI || del_t > 1) {
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
                
                // Store the image timestamp
                image_timestamps.insert({ image_name, timestamp });
            }
        }
        gt_file.close();

        return true;
    }

    bool loadImages() override
    {
        return LoadImages_(this->image_dir_, this->images_, this->cameras_data_, this->image_timestamps, this->gps_datas, this->sfm_data_);
    }

    bool LoadImages_
    (
        const std::string& image_dir,
        const std::vector<std::string>& images,
        const std::vector<openMVG::cameras::PinholeCamera>& cameras,
        std::map<std::string, double>& image_timestamps,
        const std::map<double, GPS_Data_ARkit>& gps_datas,
        openMVG::sfm::SfM_Data& sfm_data
    )
    {
        if (image_dir.empty() || !stlplus::is_folder(image_dir))
        {
            OPENMVG_LOG_ERROR << "Invalid input image directory";
            return false;
        }
        if (images.empty())
        {
            OPENMVG_LOG_ERROR << "Invalid input image sequence";
            return false;
        }
        if (cameras.empty())
        {
            OPENMVG_LOG_ERROR << "Invalid input camera data";
            return false;
        }
        if (image_timestamps.empty())
        {
            OPENMVG_LOG_ERROR << "Invalid input image timestamps";
            return false;
        }
        if (gps_datas.empty())
        {
            OPENMVG_LOG_ERROR << "Invalid input gps data";
            return false;
        }

        sfm_data.s_root_path = image_dir; // Setup main image root_path

        Views& views = sfm_data.views;
        Poses& poses = sfm_data.poses;
        Intrinsics& intrinsics = sfm_data.intrinsics;

        system::LoggerProgress my_progress_bar(images.size(), "- Loading dataset images -");
        std::ostringstream error_report_stream;
        auto iter_camera = cameras.cbegin();
        for (auto iter_image = images.cbegin();
            iter_image != images.cend();
            ++iter_image, ++iter_camera, ++my_progress_bar)
        {
            const std::string sImageFilename = stlplus::create_filespec(image_dir, *iter_image);
            const std::string sImFilenamePart = stlplus::filename_part(sImageFilename);

            OPENMVG_LOG_INFO << "Loading image : " << sImageFilename << std::endl;

            // find gps reading with closest timestamp
            double timestamp = image_timestamps[sImFilenamePart];
            double min_diff = 1000000000;
            double time_limit = 0.2;
            double closest_gps_reading = -1.0;
            std::shared_ptr<double> closest_gps_reading_ptr=nullptr;
            for ( const auto &gps_ : gps_datas ) {
                double gps_timestamp = gps_.first;
                double diff = fabs(gps_timestamp - timestamp);
                if (diff < min_diff && diff < time_limit){
                    min_diff = diff;
                    closest_gps_reading = gps_timestamp;
                }
            }
                
            GPS_Data_ARkit gps_reading;
            if (closest_gps_reading != -1.0){
                 gps_reading= gps_datas.at(closest_gps_reading);
            }



            // Test if the image format is supported
            if (openMVG::image::GetFormat(sImageFilename.c_str()) == openMVG::image::Unknown)
            {
                error_report_stream
                    << sImFilenamePart << ": Unkown image file format." << "\n";
                continue; // Image cannot be opened
            }

            if (sImFilenamePart.find("mask.png") != std::string::npos
                || sImFilenamePart.find("_mask.png") != std::string::npos)
            {
                error_report_stream
                    << sImFilenamePart << " is a mask image" << "\n";
                continue;
            }

            // Test if this image can be read
            openMVG::image::ImageHeader imgHeader;
            if (!openMVG::image::ReadImageHeader(sImageFilename.c_str(), &imgHeader))
                continue; // Image cannot be read

            const Mat3 K = iter_camera->_K;
            const double focal = (K(0, 0) + K(1, 1)) / 2.0; //Assume K(0,0)==K(1,1)
            const double pxx = K(0, 2);
            const double pyy = K(1, 2);

            const Pose3 pose(iter_camera->_R, iter_camera->_C);

            // const auto view = std::make_shared<sfm::ViewPriors>(*iter_image, views.size(), views.size(), views.size(), imgHeader.width, imgHeader.height);

            // if (closest_gps_reading != -1.0){
            //     view->b_use_pose_center_ = true;
            //     view->pose_center_ = Vec3(gps_reading.latitude_, gps_reading.longitude_, gps_reading.altitude_);
            //     view->center_weight_ = Vec3(1.0, 1.0, 1.0);
                
            // }
                        
            const auto intrinsic = std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(
                imgHeader.width, imgHeader.height,
                focal, pxx, pyy);
            
            
            if (closest_gps_reading != -1.0){
                sfm::ViewPriors view(*iter_image, views.size(), views.size(), views.size(), imgHeader.width, imgHeader.height);
                view.SetPoseCenterPrior(geodesy::lla_to_utm(gps_reading.latitude_, gps_reading.longitude_, gps_reading.altitude_),
                                         Vec3(1.0, 1.0, 1.0));
                // Add the view to the sfm_container
                views[view.id_view] = std::make_shared<sfm::ViewPriors>(view);
                // Add the pose to the sfm_container
                poses[view.id_pose] = pose;
                // Add the intrinsic to the sfm_container
                intrinsics[view.id_intrinsic] = intrinsic;
            }
            else{
                sfm::View view(*iter_image, views.size(), views.size(), views.size(), imgHeader.width, imgHeader.height);
                // Add the view to the sfm_container
                views[view.id_view] = std::make_shared<sfm::View>(view);
                // Add the pose to the sfm_container
                poses[view.id_pose] = pose;
                // Add the intrinsic to the sfm_container
                intrinsics[view.id_intrinsic] = intrinsic;
            }            
        }

        // Display saved warning & error messages if any.
        if (!error_report_stream.str().empty())
        {
            OPENMVG_LOG_ERROR
                << "\nWarning & Error messages:\n"
                << error_report_stream.str() << std::endl;
        }

        // Group the camera that share the same set of camera parameters
        GroupSharedIntrinsics(sfm_data);

        return true;
    }

};

#endif // IO_READ_GT_ARKIT_HPP
