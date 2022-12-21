
#ifndef SLS_PUT_HEADER_BMP
#define SLS_PUT_HEADER_BMP

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "bmp.h"

#ifndef SLS_FPUTC2LH
#define SLS_FPUTC2LH

int fputc2lh(unsigned short int d, FILE *sput)
{
  if (putc(d & 0xFF, sput) == EOF || putc(d >> CHAR_BIT, sput) == EOF) {
    return EOF;
  }
  return 0;
}

#endif

#ifndef SLS_FPUTC4LH
#define SLS_FPUTC4LH

int fputc4lh(unsigned long int d, FILE *sput)
{
  if (putc(d & 0xFF, sput) == EOF || putc((d >> CHAR_BIT) & 0xFF, sput) == EOF ||
      putc((d >> CHAR_BIT * 2) & 0xFF, sput) == EOF || putc((d >> CHAR_BIT * 3) & 0xFF, sput) == EOF) {
    return EOF;
  }
  return 0;
}

#endif

int put_header_bmp(FILE *sput, int rx, int ry, int cbit)
{
  int i, nx, color;
  unsigned long int bfOffBits;

  if (rx <= 0 || ry <= 0) {
    return IFR_SIZE_ERR;
  }
  if (sput == NULL || ferror(sput)) {
    return IFR_PUT_ERR;
  }
  nx = rx - (rx % 4);
  if (nx < rx) {
    nx += 4;
  }

  if (cbit == 24) {
    color = 0;
  } else {
    color = 1;
    for (i = 1; i <= cbit; i++) {
      color *= 2;
    }
  }
  bfOffBits = 14 + 40 + 4 * color;

  if (fputs("BM", sput) == EOF || fputc4lh(bfOffBits + (unsigned long) nx * ry, sput) == EOF ||
      fputc2lh(0, sput) == EOF || fputc2lh(0, sput) == EOF || fputc4lh(bfOffBits, sput) == EOF ||
      ferror(sput)) {
    return IFR_PUT_ERR;
  }

  if (fputc4lh(40, sput) == EOF || fputc4lh(rx, sput) == EOF || fputc4lh(ry, sput) == EOF ||
      fputc2lh(1, sput) == EOF || fputc2lh(cbit, sput) == EOF || fputc4lh(0, sput) == EOF || fputc4lh(0, sput) == EOF || fputc4lh(0, sput) == EOF ||
      fputc4lh(0, sput) == EOF || fputc4lh(0, sput) == EOF || fputc4lh(0, sput) == EOF || ferror(sput)) {
    return IFR_PUT_ERR;
  }

  return IFR_SUCCESS;
}

#endif

int put_header_bmp(FILE *sput, int nx, int ny, int cbit);


void DFR8bmp(char* data, int width, int height, char* filename, char* palette)
{
    unsigned char pal[768], rgba[4] = {0,0,0,0}, *padded_data;
    int mx, jx, jy;
    FILE *fp, *fc;

    fp = fopen(palette, "rb");
    if (fp != NULL) {
        fread(pal, 1, 768, fp);
        fclose(fp);
    }

    fc = fopen(filename, "wb");
    put_header_bmp(fc, width, height, 8);

    for (int j = 0; j < 256; j++) {
        rgba[0] = pal[3 * j + 2];
        rgba[1] = pal[3 * j + 1];
        rgba[2] = pal[3 * j + 0];
        fwrite(rgba, 1, 4, fc);
    }

    mx = width - (width % 4);
    if (mx < width) {
        mx += 4;
    }

    padded_data = (unsigned char*) calloc(mx * height, sizeof(unsigned char));
    for (jy = 0; jy < height; jy++) {
        for (jx = 0; jx < width; jx++) {
            padded_data[jy * mx + jx] = data[(height - 1 - jy) * width + jx];
        }
    }
    fwrite(padded_data, 1, mx * height, fc);

    fclose(fc);
    free(padded_data);
}
