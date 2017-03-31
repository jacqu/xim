// xim : basic framework to handle XIMEA cameras with RPIt
// JG, 30.03.17

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <termios.h>
#include <memory.h>
#include <m3api/xiApi.h>
#include <opencv2/core.hpp> 
#include <opencv2/highgui/highgui.hpp>

#define HandleResult(res,place) if (res!=XI_OK) {printf("Error after %s (%d).\n",place,res);goto finish;}

#define XIM_LIVE_VIDEO										// Live video on/off
#define XIM_VIDEO_FSKIP			30						// Live video frame subsampling
#define XIM_EXPOSURE				1000					// us
#define XIM_TIMEOUT					5000					// ms
#define XIM_VIDEO_NAME			"Live video"	// Name of the live video window

using namespace cv;
using namespace std;

// 
// getch and kbhit
// 
// Helper functions for interactions in console.
// 
 
int getch(void)
{
  struct termios oldt, newt;
  int ch;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return ch;
}

int kbhit(void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF)
  {
    ungetc(ch, stdin);
    return 1;
  }

  return 0;
}

// Compute padding
int xim_get_padding_x( XI_IMG* xiimg )
{
	switch( xiimg->frm )
	{
	case XI_RAW16:
	case XI_MONO16:
		return xiimg->padding_x/2;
	default:
		return xiimg->padding_x;	
	}	
}

// Converts a XiAPI image to OpenCV IplImage.
// Creates CV image.
// Return NULL if error, valid pointer otherwise.
IplImage* xim_xiimg2cvipl( XI_IMG* xiimg )	{
	IplImage* cvipl;
	
	// Create new IPl Image according to XI_IMG characteristics
	switch( xiimg->frm )
		{
		case XI_MONO8       : cvipl = cvCreateImage( cvSize( xiimg->width + xim_get_padding_x( xiimg ), xiimg->height ), IPL_DEPTH_8U, 1 ); break;
		case XI_RAW8        : cvipl = cvCreateImage( cvSize( xiimg->width, xiimg->height ), IPL_DEPTH_8U, 1 ); break;		
		case XI_MONO16      :
		case XI_RAW16       : cvipl = cvCreateImage( cvSize( xiimg->width, xiimg->height ), IPL_DEPTH_16U, 1 ); break;
		case XI_RGB24       :
		case XI_RGB_PLANAR  : cvipl = cvCreateImage( cvSize( xiimg->width, xiimg->height ), IPL_DEPTH_8U, 3 );	break;
		case XI_RGB32       : cvipl = cvCreateImage( cvSize( xiimg->width, xiimg->height ), IPL_DEPTH_8U, 4 ); break;
		default :
			printf( "xim_xiimg2cvipl error: unknown format.\n");
			return NULL;
		}
	
	// Defines the pointer to image data
	cvipl->imageData=(char*)xiimg->bp;

	return cvipl;
}

int main( int argc, char* argv[] )
{
	XI_IMG 		image;
	IplImage*	cv_image;
	cv::Mat		cv_mat;
	int				frame_cnt = 0;
	
	// Initialize image buffer
	memset( &image, 0, sizeof(image) );
	image.size = sizeof( XI_IMG );

	// XIMEA API V4.05
	HANDLE xiH = NULL;
	XI_RETURN stat = XI_OK;

	// Retrieving a handle to the camera device 
	printf( "Opening first camera.\n" );
	stat = xiOpenDevice( 0, &xiH );
	HandleResult( stat, "xiOpenDevice" );

	// Setting "exposure" parameter (us)
	stat = xiSetParamInt( xiH, XI_PRM_EXPOSURE, XIM_EXPOSURE );
	HandleResult( stat,"xiSetParam (exposure set)" );

	// Starting acquisition
	printf( "Starting acquisition.\n" );
	stat = xiStartAcquisition( xiH );
	HandleResult( stat, "xiStartAcquisition" );

	// Get first image to detect format
	stat = xiGetImage( xiH, XIM_TIMEOUT, &image );
	HandleResult( stat, "xiGetImage" );

	// Initialize OpenCV
	cv_image = xim_xiimg2cvipl( &image );
	if ( cv_image == NULL )	{
		printf( "Error after xim_xiimg2cvipl (NULL).\n" );
		goto finish;
	}
	cv_mat = cv::cvarrToMat( cv_image );

	// Create video window
	#ifdef XIM_LIVE_VIDEO
	cv::namedWindow( XIM_VIDEO_NAME, WINDOW_AUTOSIZE );
	#endif

	// Acquisition loop
	while( 1 )
	{
		// Getting image from camera
		stat = xiGetImage( xiH, XIM_TIMEOUT, &image );
		HandleResult( stat, "xiGetImage" );
		
		// Update CV image
		cv_image->imageData=(char*)image.bp;
		cv_mat = cv::cvarrToMat( cv_image );
		
		#ifdef XIM_LIVE_VIDEO
		// Display image at desired fps
		if ( !( frame_cnt++ % XIM_VIDEO_FSKIP ) )	{
			cv::imshow( XIM_VIDEO_NAME, cv_mat );
			// Exit if key pressed + let the time for the image to be displayed
			if ( cvWaitKey( XIM_EXPOSURE / 1000 ) != -1 )
				break;
			}
		#else
		// Exit if key pressed
		if ( kbhit( ) )
			break;
		#endif
	}
	
	// Shortcut in case of error
	finish:
	
	// Cleanup
	printf( "Stopping acquisition.\n" );
	#ifdef XIM_LIVE_VIDEO
	destroyWindow( XIM_VIDEO_NAME );
	#endif
	cvReleaseImage(	&cv_image	);
	xiStopAcquisition( xiH );
	xiCloseDevice( xiH );

	printf("Done.\n");

	return 0;
}

