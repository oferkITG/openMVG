// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2016 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/geometry/rigid_transformation3D_srt.hpp"
#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/geometry/Similarity3_Kernel.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_data_transform.hpp"

#include "openMVG/exif/exif_IO_EasyExif.hpp"
#include "openMVG/geodesy/geodesy.hpp"

// //- Robust estimation - LMeds (since no threshold can be defined)
#include "openMVG/robust_estimation/robust_estimator_LMeds.hpp"

#include "software/SfM/SfMPlyHelper.hpp"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "third_party/cmdLine/cmdLine.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>

using namespace openMVG;
using namespace openMVG::exif;
using namespace openMVG::geodesy;
using namespace openMVG::sfm;

struct GPS_data{
    double timestamp_;
    int gps_id;
    int frame_id;
    std::vector<double> parameter_;
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

            std::istringstream line_stream(line);
            std::getline(line_stream, substring, ',');
            tmp_gps.timestamp_ = stod(substring);
            std::getline(line_stream, substring, ',');
            tmp_gps.gps_id = stoi(substring);
            
            while (std::getline(line_stream, substring, ',')) {

                tmp_gps.parameter_.push_back(stod(substring));
            }

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
            std::cout << tmp_gps.frame_id << " " << std::to_string(tmp_gps.timestamp_) << std::endl;
        }
        gps_data_file.close();

  return true;
}

int main(int argc, char **argv)
{
  enum ERegistrationType
  {
    ROBUST_RIGID_REGISTRATION = 0,
    RIGID_REGISTRATION_ALL_POINTS = 1
  };
  std::string
    sSfM_Data_Dir_In,
    sSfM_Data_Filename_In,
    sSfM_Data_Filename_Out;
  unsigned int rigid_registration_method = ERegistrationType::RIGID_REGISTRATION_ALL_POINTS;
  int i_GPS_XYZ_method = 0;
  CmdLine cmd;
  cmd.add(make_option('d', sSfM_Data_Dir_In, "input_folder"));
  cmd.add(make_option('i', sSfM_Data_Filename_In, "input_file"));
  cmd.add(make_option('o', sSfM_Data_Filename_Out, "output_file"));
  cmd.add(make_option('m', rigid_registration_method, "method"));
  cmd.add(make_option('M', i_GPS_XYZ_method, "gps_to_xyz_method"));
  try
  {
    if (argc == 1) throw std::string("Invalid command line parameter.");
    cmd.process(argc, argv);
  }
  catch (const std::string& s)
  {
    std::cerr
      << "Usage: " << argv[0] << '\n'
      << " GPS registration of a SfM Data scene,\n"
      << "[-d|--input_folder] path to the input SfM_Data scene folder\n"
      << "[-i|--input_file] path to the input SfM_Data scene\n"
      << "[-o|--output_file] path to the output SfM_Data scene\n"
      << "[-m|--method] method to use for the rigid registration\n"
      << "\t0 => registration is done using a robust estimation,\n"
      << "\t1 (default)=> registration is done using all points.\n"
	  << "[-M|--gps_to_xyz_method] XZY Coordinate system:\n"
	  << "\t 0: ECEF (default)\n"
	  << "\t 1: UTM\n"
    << "\t 2: XYZ"
      << std::endl;

    std::cerr << s << std::endl;
    return EXIT_FAILURE;
  }

  if (sSfM_Data_Filename_In.empty() || sSfM_Data_Filename_Out.empty())
  {
    std::cerr << "Invalid input or output filename." << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Load a SfM scene
  // For each valid view (pose & intrinsic defined)
  //  - iff a GPS position can be parsed
  //    - store corresponding camera pose & GPS position
  // - Compute the registration between the selected camera poses & GPS positions

  // Load input SfM_Data scene
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename_In, ESfM_Data(ALL)))
  {
    std::cerr
      << "\nThe input SfM_Data file \"" << sSfM_Data_Filename_In
      << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }

  std::map<int,GPS_data> gps_data_list;
  if(!load_GPS_data(sSfM_Data_Dir_In, sfm_data, gps_data_list))
  {
    std::cerr
      << "\nThe input GPS_Data couldn't be read" << std::endl;
    return EXIT_FAILURE;
  }

  // List corresponding poses (SfM - GPS)
  std::vector<Vec3> vec_sfm_center, vec_gps_center;

  for (const auto & view_it : sfm_data.GetViews() )
  {
    if (!sfm_data.IsPoseAndIntrinsicDefined(view_it.second.get()))
      continue;

    const std::string view_filename =
      stlplus::create_filespec(sfm_data.s_root_path, view_it.second->s_Img_path);

    double latitude, longitude, altitude;

    // Check existence of GPS coordinates
    if(gps_data_list.find(view_it.second->id_view) != gps_data_list.end()) {
      GPS_data gps_data = gps_data_list[view_it.second->id_view];
      latitude = gps_data.parameter_[0];
      altitude = gps_data.parameter_[1];
      longitude = gps_data.parameter_[2];

      // Add XYZ position to the GPS position array
      switch (i_GPS_XYZ_method)
      {
      case 2:
      {
        openMVG::Vec3 gps_center;
        gps_center.x() = latitude;
        gps_center.y() = altitude;
        gps_center.z() = longitude;

        vec_gps_center.push_back(gps_center);
        break;
      }
      case 1:
        vec_gps_center.push_back(lla_to_utm(latitude, longitude, altitude));
        break;
      case 0:
      default:
        vec_gps_center.push_back(lla_to_ecef(latitude, longitude, altitude));
        break;
      }
      const openMVG::geometry::Pose3 pose(sfm_data.GetPoseOrDie(view_it.second.get()));
      vec_sfm_center.push_back( pose.center() );
    }

  }

  if ( vec_sfm_center.empty() )
  {
    std::cerr << "No valid corresponding GPS data found for the used views." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::endl
    << "Registration report:\n"
    << " #corresponding SFM - GPS data: " << vec_sfm_center.size() << "\n"
    << std::endl;

  // Export the corresponding poses (for debugging & see the transformation)
  // plyHelper::exportToPly( vec_gps_center,
  //   stlplus::create_filespec(stlplus::folder_part(sSfM_Data_Filename_Out), "GPS_position", "ply"));
  // plyHelper::exportToPly( vec_sfm_center,
  //   stlplus::create_filespec(stlplus::folder_part(sSfM_Data_Filename_Out), "SFM_position", "ply"));

  {
    // Convert positions to the appropriate data container
    const Mat X_SfM = Eigen::Map<Mat>(vec_sfm_center[0].data(), 3, vec_sfm_center.size());
    const Mat X_GPS = Eigen::Map<Mat>(vec_gps_center[0].data(), 3, vec_gps_center.size());

    openMVG::geometry::Similarity3 sim;

    // Compute the registration:
    // - using a rigid scheme (using all points)
    // - using a robust scheme (using partial points - robust estimation)
    switch (rigid_registration_method)
    {
      case ERegistrationType::ROBUST_RIGID_REGISTRATION:
      {
        using namespace openMVG::robust;
        using namespace openMVG::geometry;

        geometry::kernel::Similarity3_Kernel kernel(X_SfM, X_GPS);
        const double lmeds_median = LeastMedianOfSquares
          (
            kernel,
            &sim
          );
        std::cout << "LMeds found a model with an upper bound of: " <<  sqrt(lmeds_median) << " user units."<< std::endl;

        // Compute & display fitting errors
        {
          const Vec vec_fitting_errors_eigen(
            geometry::kernel::Similarity3ErrorSquaredMetric::ErrorVec(sim, X_SfM, X_GPS).array().sqrt());
          std::cout << "\n3D Similarity fitting error using all points (in target coordinate system units):";
          minMaxMeanMedian<float>(
            vec_fitting_errors_eigen.data(),
            vec_fitting_errors_eigen.data() + vec_fitting_errors_eigen.rows(), std::cout);
        }
        // INLIERS only
        {
          std::vector<float> vec_fitting_errors;
          for (Mat::Index i = 0; i < X_SfM.cols(); ++i)
          {
            if (geometry::kernel::Similarity3ErrorSquaredMetric::Error(sim, X_SfM.col(i), X_GPS.col(i)) < lmeds_median)
              vec_fitting_errors.push_back((X_GPS.col(i) - sim(X_SfM.col(i))).norm());
          }
          std::cout << "\nFound: " << vec_fitting_errors.size() << " inliers"
           << " from " << X_SfM.cols() << " points." << std::endl;
          std::cout << "\n3D Similarity fitting error using only the fitted inliers (in target coordinate system units):";
          minMaxMeanMedian<float>( vec_fitting_errors.cbegin(), vec_fitting_errors.cend(), std::cout);
        }
      }
      break;
      case ERegistrationType::RIGID_REGISTRATION_ALL_POINTS:
      {
        Vec3 t;
        Mat3 R;
        double S;
        if (!openMVG::geometry::FindRTS(X_SfM, X_GPS, &S, &t, &R))
        {
          std::cerr << "Failed to compute the registration" << std::endl;
          return EXIT_FAILURE;
        }

        std::cout
          << "Found transform:\n"
          << " scale: " << S << "\n"
          << " rotation:\n" << R << "\n"
          << " translation: " << std::fixed << std::setprecision(9)
          << t.transpose() << std::endl;

        // Encode the transformation as a 3D Similarity transformation matrix // S * R * X + t
        sim = openMVG::geometry::Similarity3(geometry::Pose3(R, -R.transpose()* t/S), S);

        // Compute & display fitting errors
        {
          const Vec vec_fitting_errors_eigen(
            geometry::kernel::Similarity3ErrorSquaredMetric::ErrorVec(sim, X_SfM, X_GPS).array().sqrt());
          std::cout << "\n3D Similarity fitting error (in target coordinate system units):";
          minMaxMeanMedian<float>(
            vec_fitting_errors_eigen.data(),
            vec_fitting_errors_eigen.data() + vec_fitting_errors_eigen.rows(),
            std::cout);
        }
      }
      break;
      default:
      std::cerr << "Unknow rigid registration method" << std::endl;
      return EXIT_FAILURE;
    }

    //--
    // Apply the found transformation to the SfM Data Scene
    //--
    openMVG::sfm::ApplySimilarity(sim, sfm_data);
  }

  // Export the SfM_Data scene in the expected format
  if (Save(
        sfm_data,
        sSfM_Data_Filename_Out.c_str(),
        ESfM_Data(ALL)))
  {
    return EXIT_SUCCESS;
  }
  else
  {
    std::cerr
      << std::endl
      << "An error occured while trying to save \""
      << sSfM_Data_Filename_Out << "\"." << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_FAILURE;
}
