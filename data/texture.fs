#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

// texture samplers
uniform sampler2D tex;

void main()
{
	// linearly interpolate between both textures (80% container, 20% awesomeface)
    float color = texture(tex, TexCoord).r;
	FragColor = vec4(color, color, color, 1.f);
}
