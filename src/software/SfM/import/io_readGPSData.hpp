#ifndef IO_LOAD_GPS_HPP
#define IO_LOAD_GPS_HPP

#include "openMVG/sfm/sfm_data.hpp"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <fstream>

using namespace openMVG;
using namespace openMVG::sfm;

struct GPS_data{
    double timestamp_;
    int gps_id;
    int frame_id;
    double lat;
    double lon;
    double alt;
};

//timestamp, frame id, flx, fly, px, py
struct Frame_Data {
    double timestamp_;
    int frame_id;
};

bool load_GPS_data(const std::string GPS_dir, const SfM_Data sfm_data, std::map<int,GPS_data>& gps_data) {

        std::map<double, Frame_Data> frame_datas;

        std::ifstream frames_data_file(stlplus::create_filespec(GPS_dir, "Frames.txt"), std::ifstream::in);
        if (!frames_data_file)
        {
            std::cerr << "Error: Failed to open file '" << stlplus::create_filespec(GPS_dir, "Frames.txt") << "' for reading" << std::endl;
            return false;
        }
        while (frames_data_file)
        {
            std::string line;
            std::getline(frames_data_file, line);
            if (line.size() == 0 || line[0] == '#' || !isdigit(line[0]))
            {
                continue;
            }

            Frame_Data temp_frame;
            std::string substring;

            std::istringstream line_stream(line);
            std::getline(line_stream, substring, ',');
            temp_frame.timestamp_ = stod(substring);
            std::getline(line_stream, substring, ',');
            temp_frame.frame_id = stoi(substring);

            frame_datas.insert({ temp_frame.timestamp_,temp_frame });
        }
        frames_data_file.close();

        std::ifstream gps_data_file(stlplus::create_filespec(GPS_dir, "Anchors.txt"), std::ifstream::in);
        if (!gps_data_file)
        {
            std::cerr << "Error: Failed to open file '" << stlplus::create_filespec(GPS_dir, "Anchors.txt") << "' for reading" << std::endl;
            return false;
        }
        while (gps_data_file)
        {
            std::string line;
            std::getline(gps_data_file, line);
            if (line.size() == 0 || line[0] == '#' || !isdigit(line[0]))
            {
                continue;
            }

            GPS_data tmp_gps;
            std::string substring;
            Mat3 K, R;
            Vec3 t;
            Eigen::Quaterniond quaternionf_rotation;

            std::istringstream image_stream(line);

            std::getline(image_stream, substring, ',');
            tmp_gps.timestamp_ = stod(substring);
            std::getline(image_stream, substring, ',');
            tmp_gps.gps_id = stod(substring);
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
            
            R = quaternionf_rotation.toRotationMatrix();
            const Mat3 transform_matrix = (Mat3() << 1, 1, 1, -1, -1, -1, -1, -1, -1).finished();
            Mat3 Rtranspose = R.transpose();
            R = transform_matrix.cwiseProduct(Rtranspose);
            
            t = -R * t;

            tmp_gps.lat = t.x();
            tmp_gps.lon = t.z();
            tmp_gps.alt = t.y();

            std::map<double, Frame_Data>::iterator frm_data_it = frame_datas.find(tmp_gps.timestamp_);
            if(frm_data_it == frame_datas.end()) {
              
              for (frm_data_it = frame_datas.begin(); frm_data_it != frame_datas.end(); frm_data_it++) {
                std::map<double, Frame_Data>::iterator tmp_frm_data_nxt = std::next(frm_data_it);
                if((tmp_gps.timestamp_ > frm_data_it->second.timestamp_ && tmp_gps.timestamp_ < tmp_frm_data_nxt->second.timestamp_)) {
                  break;
                }
              }

              assert(frm_data_it != frame_datas.end());
              
              std::map<double, Frame_Data>::iterator frm_data_nxt = std::next(frm_data_it);
              if((tmp_gps.timestamp_ - frm_data_it->second.timestamp_) < (tmp_gps.timestamp_ - frm_data_nxt->second.timestamp_)) 
                frm_data_it = frm_data_it;
              else
                frm_data_it = frm_data_nxt;
            }
            
            Frame_Data frm_data = frm_data_it->second;
            tmp_gps.frame_id = frm_data.frame_id;

            gps_data.insert({ tmp_gps.frame_id,tmp_gps });
            //std::cout << tmp_gps.frame_id << " " << std::to_string(tmp_gps.timestamp_) << std::endl;
        }
        gps_data_file.close();

  return true;
}

#endif // IO_LOAD_GPS_HPP