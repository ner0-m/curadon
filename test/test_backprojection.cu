#include "doctest/doctest.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "curadon/backprojection.hpp"
#include "curadon/bmp.hpp"

void draw(float *data, int slice, int width, int height, int depth);

curad::geometry setup_geom(size_t volsize, size_t width, size_t height, float angle, float DSO,
                           float DSD) {
    curad::geometry geo;
    geo.DSO = DSO;
    geo.DSD = DSD;

    geo.det_shape_ = curad::Vec<std::uint64_t, 2>{width, height};
    geo.det_spacing_ = curad::Vec<float, 2>{1, 1};
    geo.det_size_ = geo.det_shape_ * geo.det_spacing_;
    geo.det_offset_ = curad::Vec<float, 2>{0, 0};
    geo.det_rotation_ = curad::Vec<float, 3>{0, 0, 0};
    geo.COR_ = 0;

    geo.vol_shape_ = curad::Vec<std::uint64_t, 3>{volsize, volsize, volsize};
    geo.vol_spacing_ = curad::Vec<float, 3>{1, 1, 1};
    geo.vol_size_ = geo.vol_shape_ * geo.vol_spacing_;

    geo.phi_ = angle * M_PI / 180;
    geo.theta_ = 0;
    geo.psi_ = 0;

    return geo;
}

TEST_CASE("test_backprojection") {

    const auto volsize = 64;
    auto [data, width, height, nangles, angles, DSO, DSD] = curad::easy::read("demofile2.txt");
    // std::cout << "Width: " << width << " Height: " << height << " nangles: " << nangles << "\n";

    CHECK_EQ(width, 16);
    CHECK_EQ(height, 16);
    CHECK_EQ(nangles, 1);
    CHECK_EQ(DSO, 100);
    CHECK_EQ(DSD, 120);

    thrust::host_vector<float> host_sino(width * height * nangles, 0);
    std::copy(data.begin(), data.end(), host_sino.begin());

    thrust::device_vector<float> sino = host_sino;
    auto sino_ptr = thrust::raw_pointer_cast(sino.data());

    thrust::device_vector<float> volume(volsize * volsize * volsize, 0);
    auto volume_ptr = thrust::raw_pointer_cast(volume.data());
    gpuErrchk(cudaDeviceSynchronize());

    auto det_shape = curad::Vec<std::uint64_t, 2>{width, height};
    auto vol_shape = curad::Vec<std::uint64_t, 3>{volsize, volsize, volsize};

    // auto vol_spacing = curad::Vec<float, 3>{1, 1, 1};
    auto vol_spacing = curad::Vec<float, 3>{3, 3, 3};
    auto vol_size = vol_shape * vol_spacing;
    auto vol_offset = curad::Vec<float, 3>{0, 0, 0};

    curad::Vec<float, 3> source({0, 0, -DSO});

    const auto stride_x = 1;
    const auto stride_y = volsize;
    const auto stride_z = volsize * volsize;

    curad::backproject_3d(volume_ptr, stride_x, stride_y, stride_z, vol_shape, vol_size,
                          vol_spacing, vol_offset, sino_ptr, width, height, det_shape, angles, source, DSD,
                          DSO);

    thrust::host_vector<float> vol_host = volume;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    std::transform(vol_host.begin(), vol_host.end(), vol_host.begin(),
                   [&](auto x) { return x / 50; });

    // std::cout << "max of vol_host: " << *std::max_element(vol_host.begin(), vol_host.end())
    //           << std::endl;
    //
    // std::cout << "max of data: " << *std::max_element(data.begin(), data.end()) << std::endl;

    // auto max_elem = *std::max_element(data.begin(), data.end());
    auto max_elem = *std::max_element(vol_host.begin(), vol_host.end());
    std::transform(vol_host.begin(), vol_host.end(), vol_host.begin(),
                   [&](auto x) { return x / max_elem; });
    const auto slice = volsize / 2;
    draw(thrust::raw_pointer_cast(vol_host.data()), slice, volsize, volsize, volsize);
    // draw(data.data(), slice, width, height, nangles);
}

// Include GLEW. Always include it before gl.h and glfw3.h, since it's a bit magic.
#include <GL/glew.h>

#include <GL/glut.h>
#include <GLFW/glfw3.h>

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window, unsigned int &curtex, unsigned int maxtex) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    } else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        curtex = (curtex + 1) % maxtex;
        // std::cout << "Current texture: " << curtex << " / " << maxtex << std::endl;
    } else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        curtex = (curtex - 1) % maxtex;
        // std::cout << "Current texture: " << curtex << " / " << maxtex << std::endl;
    }
}

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

class Shader {
  public:
    unsigned int ID;
    // constructor generates the shader on the fly
    // ------------------------------------------------------------------------
    Shader(const char *vertexPath, const char *fragmentPath) {
        // 1. retrieve the vertex/fragment source code from filePath
        std::string vertexCode;
        std::string fragmentCode;
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;
        // ensure ifstream objects can throw exceptions:
        vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try {
            // open files
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
            std::stringstream vShaderStream, fShaderStream;
            // read file's buffer contents into streams
            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();
            // close file handlers
            vShaderFile.close();
            fShaderFile.close();
            // convert stream into string
            vertexCode = vShaderStream.str();
            fragmentCode = fShaderStream.str();
        } catch (std::ifstream::failure &e) {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
        }
        const char *vShaderCode = vertexCode.c_str();
        const char *fShaderCode = fragmentCode.c_str();
        // 2. compile shaders
        unsigned int vertex, fragment;
        // vertex shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");
        // fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");
        // shader Program
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");
        // delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }
    // activate the shader
    // ------------------------------------------------------------------------
    void use() { glUseProgram(ID); }
    // utility uniform functions
    // ------------------------------------------------------------------------
    void setBool(const std::string &name, bool value) const {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
    }
    // ------------------------------------------------------------------------
    void setInt(const std::string &name, int value) const {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
    }
    // ------------------------------------------------------------------------
    void setFloat(const std::string &name, float value) const {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
    }

  private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(unsigned int shader, std::string type) {
        int success;
        char infoLog[1024];
        if (type != "PROGRAM") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                          << infoLog
                          << "\n -- --------------------------------------------------- -- "
                          << std::endl;
            }
        } else {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                          << infoLog
                          << "\n -- --------------------------------------------------- -- "
                          << std::endl;
            }
        }
    }
};

void draw(float *data, int slice, int width, int height, int depth) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);

    glViewport(0, 0, 800, 600);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // start GLEW extension handler
    glewExperimental = GL_TRUE;
    glewInit();

    // build and compile our shader program
    // ------------------------------------
    // vertex shader
    Shader ourShader("texture.vs", "texture.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // clang-format off
    float vertices[] = {
        // positions          // colors           // texture coords
         1.f,  1.f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
         1.f, -1.f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.f, -1.f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -1.f,  1.f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    // clang-format on

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // load and create a texture
    // -------------------------
    std::vector<unsigned int> textures;
    textures.reserve(depth);
    for (int i = 0; i < depth; ++i) {
        // texture 1
        // ---------
        unsigned int tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // load image, create texture and generate mipmaps
        const auto offset = i * width * height;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, data + offset);

        textures.push_back(tex);
    }

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done
    // once)
    // -------------------------------------------------------------------------------------------
    ourShader.use(); // don't forget to activate/use the shader before setting uniforms!
    // or set it via the texture class
    ourShader.setInt("texture1", 0);

    unsigned int curtex = 0;
    while (!glfwWindowShouldClose(window)) {
        processInput(window, curtex, textures.size());

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures[curtex]);

        // render container
        ourShader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwTerminate();
}
