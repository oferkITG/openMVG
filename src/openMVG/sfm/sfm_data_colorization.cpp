// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "openMVG/sfm/sfm_data_colorization.hpp"

#include "openMVG/image/image_container.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/image/pixel_types.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/loggerprogress.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

namespace openMVG {
namespace sfm {

/// Find the color of the SfM_Data Landmarks/structure
bool ColorizeTracks(
  const SfM_Data & sfm_data,
  std::vector<Vec3> & vec_3dPoints,
  std::vector<Vec3> & vec_tracksColor,
  const std::string & sSfM_Images_path)
{
  // Colorize each track
  // Start with the most representative image
  //   and iterate to provide a color to each 3D point

  {
    system::LoggerProgress my_progress_bar(sfm_data.GetLandmarks().size(),"- Compute scene structure color -" );

    vec_tracksColor.resize(sfm_data.GetLandmarks().size());
    vec_3dPoints.resize(sfm_data.GetLandmarks().size());

    //Build a list of contiguous index for the trackIds
    std::map<IndexT, IndexT> trackIds_to_contiguousIndexes;
    IndexT cpt = 0;
    for (Landmarks::const_iterator it = sfm_data.GetLandmarks().begin();
      it != sfm_data.GetLandmarks().end(); ++it, ++cpt)
    {
      trackIds_to_contiguousIndexes[it->first] = cpt;
      vec_3dPoints[cpt] = it->second.X;
    }

    // The track list that will be colored (point removed during the process)
    std::set<IndexT> remainingTrackToColor;
    std::transform(sfm_data.GetLandmarks().begin(), sfm_data.GetLandmarks().end(),
      std::inserter(remainingTrackToColor, remainingTrackToColor.begin()),
      stl::RetrieveKey());

    while ( !remainingTrackToColor.empty() )
    {
      // Find the most representative image (for the remaining 3D points)
      //  a. Count the number of observation per view for each 3Dpoint Index
      //  b. Sort to find the most representative view index

      std::map<IndexT, IndexT> map_IndexCardinal; // ViewId, Cardinal
      for (const auto & track_to_color_it : remainingTrackToColor)
      {
        const auto trackId = track_to_color_it;
        const Observations & obs = sfm_data.GetLandmarks().at(trackId).obs;
        for (const auto & obs_it : obs)
        {
          const auto viewId = obs_it.first;
          if (map_IndexCardinal.find(viewId) == map_IndexCardinal.end())
            map_IndexCardinal[viewId] = 1;
          else
            ++map_IndexCardinal[viewId];
        }
      }

      // Find the View index that is the most represented
      std::vector<IndexT> vec_cardinal;
      std::transform(map_IndexCardinal.begin(),
        map_IndexCardinal.end(),
        std::back_inserter(vec_cardinal),
        stl::RetrieveValue());
      using namespace stl::indexed_sort;
      std::vector<sort_index_packet_descend<IndexT, IndexT>> packet_vec(vec_cardinal.size());
      sort_index_helper(packet_vec, &vec_cardinal[0], 1);

      // First image index with the most of occurrence
      std::map<IndexT, IndexT>::const_iterator iterTT = map_IndexCardinal.begin();
      std::advance(iterTT, packet_vec[0].index);
      const size_t view_index = iterTT->first;
      const View * view = sfm_data.GetViews().at(view_index).get();
      const std::string sView_filename = stlplus::create_filespec(sfm_data.s_root_path,
        view->s_Img_path);

      std::string image_file_path = sSfM_Images_path + "//" + sView_filename.c_str();
      
      image::Image<image::RGBColor> image_rgb;
      image::Image<unsigned char> image_gray;
      const bool b_rgb_image = ReadImage(image_file_path.c_str(), &image_rgb);
      if (!b_rgb_image) //try Gray level
      {
        const bool b_gray_image = ReadImage(image_file_path.c_str(), &image_gray);
        if (!b_gray_image)
        {
          OPENMVG_LOG_ERROR << "Cannot open provided the image.";
          return false;
        }
      }

      // Iterate through the remaining track to color
      // - look if the current view is present to color the track
      std::set<IndexT> set_toRemove;
      for (const auto & track_to_color_it : remainingTrackToColor)
      {
        const auto trackId = track_to_color_it;
        const Observations & obs = sfm_data.GetLandmarks().at(trackId).obs;
        Observations::const_iterator it = obs.find(view_index);

        if (it != obs.end())
        {
          // Color the track
          const Vec2 & pt = it->second.x;
          const image::RGBColor color =
            b_rgb_image
            ? image_rgb(pt.y(), pt.x())
            : image::RGBColor(image_gray(pt.y(), pt.x()));

          vec_tracksColor[trackIds_to_contiguousIndexes.at(trackId)] =
            Vec3(color.r(), color.g(), color.b());
          set_toRemove.insert(trackId);
          ++my_progress_bar;
        }
      }
      // Remove colored track
      for (const auto & to_remove_it : set_toRemove)
      {
        remainingTrackToColor.erase(to_remove_it);
      }
    }
  }
  return true;
}

} // namespace sfm
} // namespace openMVG
