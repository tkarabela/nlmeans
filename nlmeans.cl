/*
    search window: (2*AX + 1) x (2*AY + 1)
*/
#ifndef AX
    #define AX 3
    #define AY AX
#endif

/*
    support: (2*SX + 1) x (2*SY + 1)
    this is also the size of gaussian mask
*/
#ifndef SX
    #define SX 2
    #define SY SX
#endif

/*
    helper macros
*/
#define _GET(arr,x,y) arr[(x)*cols + (y)]
#define _MASK(dx,dy) mask[(dx+SX)*(2*SY+1) + dy + SY]
#define EXPF(x) native_exp(x)

/*
    gaussian weighted SSD of supports of points (x0,y0) and (x1,y1)
*/
float
WSSD(global const float *data,
     constant const float *mask,
     int x0, int y0,
     int x1, int y1,
     int rows, int cols)
{
    float sum = 0.0f;

    for (int dx = -SX; dx <= SX; dx++)
    for (int dy = -SY; dy <= SY; dy++)
    {
        float diff = _GET(data, x0+dx, y0+dy) - _GET(data, x1+dx, y1+dy);
        float val = diff*diff * _MASK(dx, dy);
        sum += val;
    }

    return sum;
}


/*
    NLMeans for one pixel
*/
float
NLMeans_one(global const float *data,
            constant const float *mask,
            int rows, int cols,
            int x0, int y0,
            float h_squared)
{
    float fsum = 0.0f;
    float sum = 0.0f;

    for (int dx = -AX; dx <= AX; dx++)
    for (int dy = -AY; dy <= AY; dy++)
    {
        int x = x0+dx;
        int y = y0+dy;

        float f = EXPF(-WSSD(data, mask, x0, y0, x, y, rows, cols) / h_squared);
        fsum += f;

        sum += _GET(data, x, y) * f;
    }

    return sum / fsum;
}


/*
    NLMeans kernel
*/
kernel void
NLMeans_kernel(global const float *data,
               global float *output,
               constant const float *mask,
               int rows, int cols,
               float h)
{
    #if 1
    int x0 = get_group_id(0); /* index of work-group */
    int xstride =  get_num_groups(0);
    int y0 = get_local_id(0); /* index within work-group */
    int ystride = get_local_size(0);
    #else
    int x = get_global_id(0);
    int y = get_global_id(1);
    #endif

    float h_squared = h*h;

    #if 1
    for (int x = x0; x < rows; x += xstride) {
        for (int y = y0; y < cols; y += ystride) {
            float val;

            if (x > SX+AX && x < rows-SX-AX && y > SY+AY && y < cols - SY-AY) {
                val = NLMeans_one(data, mask, rows, cols, x, y, h_squared);
            } else {
                val = _GET(data, x, y);
            }

            _GET(output, x, y) = val;
        }
    }
    #else
    float val;

    if (x > SX+AX && x < rows-SX-AX && y > SY+AY && y < cols - SY-AY) {
        val = NLMeans_one(data, mask, rows, cols, x, y, h_squared);
    } else {
        val = _GET(data, x, y);
    }

    _GET(output, x, y) = val;

    #endif
}
