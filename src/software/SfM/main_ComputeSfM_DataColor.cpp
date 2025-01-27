// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_data_colorization.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/types.hpp"

#include "software/SfM/SfMPlyHelper.hpp"
#include "third_party/cmdLine/cmdLine.h"


using namespace openMVG;
using namespace openMVG::sfm;

/// Export camera poses positions as a Vec3 vector
void GetCameraPositions(const SfM_Data & sfm_data, std::vector<Vec3> & vec_camPosition)
{
  for (const auto & view : sfm_data.GetViews())
  {
    if (sfm_data.IsPoseAndIntrinsicDefined(view.second.get()))
    {
      const geometry::Pose3 pose = sfm_data.GetPoseOrDie(view.second.get());
      vec_camPosition.push_back(pose.center());
    }
  }
}

// Convert from a SfM_Data format to another
int main(int argc, char **argv)
{
  CmdLine cmd;

  std::string
    sSfM_Images_path_In,
    sSfM_Data_Filename_In,
    sOutputPLY_Out;

  cmd.add(make_option('m', sSfM_Images_path_In, "image_path"));
  cmd.add(make_option('i', sSfM_Data_Filename_In, "input_file"));
  cmd.add(make_option('o', sOutputPLY_Out, "output_file"));

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch (const std::string& s) {
      OPENMVG_LOG_INFO << "Usage: " << argv[0] << '\n'
        << "[-m|--image_path] path to the input images\n"
        << "[-i|--input_file] path to the input SfM_Data scene\n"
        << "[-o|--output_file] path to the output PLY file";

      OPENMVG_LOG_ERROR << s;
      return EXIT_FAILURE;
  }

  if (sOutputPLY_Out.empty())
  {
    OPENMVG_LOG_ERROR << "No output PLY filename specified.";
    return EXIT_FAILURE;
  }

  // Load input SfM_Data scene
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename_In, ESfM_Data(ALL)))
  {
    OPENMVG_LOG_ERROR << "The input SfM_Data file \"" << sSfM_Data_Filename_In << "\" cannot be read.";
    return EXIT_FAILURE;
  }

  // Compute the scene structure color
  std::vector<Vec3> vec_3dPoints, vec_tracksColor, vec_camPosition;
  if (ColorizeTracks(sfm_data, vec_3dPoints, vec_tracksColor, sSfM_Images_path_In))
  {
    GetCameraPositions(sfm_data, vec_camPosition);

    // Export the SfM_Data scene in the expected format
    if (plyHelper::exportToPly(vec_3dPoints, vec_camPosition, sOutputPLY_Out, &vec_tracksColor))
    {
      return EXIT_SUCCESS;
    }
  }

  return EXIT_FAILURE;
}
