/*
 * =====================================================================================
 *
 *       Filename:  glfw_app.hpp
 *
 *    Description:  glfw windowed application
 *
 *        Version:  1.0
 *        Created:  06/11/2014 20:32:52
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Pablo Colapinto (), gmail -> wolftype
 *
 * =====================================================================================
 */

#ifndef  glfw_app_INC
#define  glfw_app_INC

#include "glfw_window.hpp" //<-- Our GLFW Window


namespace lynda {

struct App{

    Window mWindow;
    Window& window() { return mWindow; }

    void App_init(int w = 1024, int h=720)  {

      /*-----------------------------------------------------------------------------
       *  Initialize GLFW
       *-----------------------------------------------------------------------------*/
      if( !glfwInit() ) exit(EXIT_FAILURE);
      printf("glfw initialized \n"); 
      
      mWindow.create(this,w,h);       //<-- Create the window, passing this application to it
      printf("glfw window created \n"); 

      /*-----------------------------------------------------------------------------
       *  Initialize GLEW
       *-----------------------------------------------------------------------------*/
      glewExperimental = true;
      GLenum glewError = glewInit();
      if (glewError != GLEW_OK){
        printf("glew init error\n%s\n", glewGetErrorString( glewError) );
      }

      if (GLEW_APPLE_vertex_array_object){
        printf("genVertexArrayAPPLE supported\n");
      } else if (GLEW_ARB_vertex_array_object){
        printf("genVertexArrays supported\n");
      }

      /*-----------------------------------------------------------------------------
       *  Some Good Defaults to Start with: enable Alpha Blending and Depth Testing
       *-----------------------------------------------------------------------------*/
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);
        
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      glLineWidth(3);               //<-- Thicken lines so we can see 'em clearly
    }


    /*-----------------------------------------------------------------------------
     *  Start the Draw Loop
     *-----------------------------------------------------------------------------*/
    void start(){
      printf("starting app\n");
      while ( !mWindow.shouldClose() ){

        mWindow.setViewport();
 
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        float * mx, *my, *mz;
        onDraw(mx, my, mz);
        
        mWindow.swapBuffers();      //<-- SWAP BUFFERS
        glfwPollEvents();           //<-- LISTEN FOR WINDOW EVENTS
       
      }

    }

    
    /*-----------------------------------------------------------------------------
     *  Properly terminate glfw when app closes
     *-----------------------------------------------------------------------------*/
    ~App(){
      glfwTerminate();
    }
    void terminate(){ glfwTerminate();}

    virtual void onDraw(float * mx, float * my, float * mz) {}

    virtual void onMouseMove(int x, int y){}

    virtual void onMouseDown(int button, int action){}

    virtual void onKeyDown(int key, int action){}

    virtual void drawOne(float * mx, float * my, float * mz){}
};

} //lynda::


#endif   /* ----- #ifndef glut_app_INC  ----- */
