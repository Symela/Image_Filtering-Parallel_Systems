#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>

extern void filtering( uint8_t* table, int height, int width, int loops, char* type, int blocksize );

// PRINT ERRORS -----------------------------------------------------------------------------------------------
void print_err(int err){
  if(err==-1)
    fprintf(stderr, "Not enough arguments-(Value of errno: %d)\n\n", err);
  else if(err==-2)
    fprintf(stderr, "Loops must be more than 0-(Value of errno: %d)\n\n", err);
  else if(err==-3)
    fprintf(stderr, "Block size must be more than 0 and smaller than the image's height and width-(Value of errno: %d)\n\n", err);
  else if(err==-4)
    fprintf(stderr, "The image type is wrong[ONLY GREY AND RGB ARE SUPPORTED]-(Value of errno: %d)\n\n", err);
  else if(err==-5)
    fprintf(stderr, "The image's dimensions do not allow it to be divided into equal blocks-(Value of errno: %d)\n\n", err);
  else if(err==-6)
    fprintf(stderr, "The image does not exist-(Value of errno: %d)\n\n", err);
  else if(err==-7)
    fprintf(stderr, "Dimensions or type of image is wrong-(Value of errno: %d)\n\n", err);
  return;
}

// CHECK ARGUMENTS -----------------------------------------------------------------------------------------------
int check_info(int argc, char** argv, int* blocksize, int* height, int* width, int* loops, char** image, char** type){
  int err=0;
  if(argc!=7){ print_err(-1); err=-1;  return err;}

  (*loops)=atoi(argv[5]);
  if((*loops)<=0){ print_err(-2); err=-2;}

  strcpy((*type),argv[2]);
  if((strcmp((*type),"RGB")!=0)&&(strcmp((*type),"GREY")!=0))
  { print_err(-4); err=-4;}

  (*height)=atoi(argv[3]);
  (*width)=atoi(argv[4]);
  (*blocksize)=atoi(argv[6]);
  if((*blocksize)<=0 || (*blocksize)>(*height) || (*blocksize)>(*width)){ print_err(-3); err=-3;}
  if(((*height)%(*blocksize)!=0)||((*width)%(*blocksize)!=0)){ print_err(-5); err=-5;}

  FILE *image_raw;
  image_raw = fopen(argv[1], "rb");
  if (!image_raw){print_err(-6); err=-6;}
  else{
    fseek(image_raw, 0L, SEEK_END);
    int size;
    size = ftell(image_raw);
    if(!strcmp((*type),"RGB")){
      if(size!=3*(*height)*(*width)){ print_err(-7); err=-7;}
      else
      {
        (*image)=(char*)malloc((size+1)*sizeof(char));
        strncpy((*image), argv[1], size);
      }
    }
    else
      if(ftell(image_raw)!=(*height)*(*width)){ print_err(-7); err=-7;}
      else
      {
        (*image)=(char*)malloc((size+1)*sizeof(char));
        strncpy((*image), argv[1], size);
      }

    fclose(image_raw);
  }

  return err;
}

// MAIN -----------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  int height, width, loops, blocksize, image_fd; // upsos, platos, epanalipseis kai fd gia to arxeio ths eikonas
  char* image, *type; // h eikona kai tupos ths: GREY or RGB
  size_t bytes;

  type = (char*)malloc(5*sizeof(char));

  if(check_info(argc, argv, &blocksize, &height, &width, &loops, &image, &type) != 0) return -1; // check ta arguments

  uint8_t *table = NULL;
  uint64_t start_time, end_time;
  struct timeval st, et;

  gettimeofday(&st, NULL);
  start_time = st.tv_sec * 1000000 + st.tv_usec;

  image_fd = open(image, O_RDONLY); // anoikse thn eikona kai diavase thn
  if(image_fd < 0)
  {
    perror("open");
    return -1;
  }

  // analoga me to type upologise to antistoixo size se bytes
  if(strcmp(type, "GREY") == 0) bytes = height * width;
  else bytes = height * width * 3;

  table = (uint8_t*) calloc(bytes, sizeof(uint8_t));
  int i, j;
  i = 0;
  j = 0;
  while(i < bytes) // apothikeuse thn eikona sthn metavlhth table
  {
    if( (j = read(image_fd, table, bytes - i)) == -1 )
    {
      perror("read");
      free(table);
    }
    i += j;
  }
  close(image_fd);

  // CUDA FUNCTIONS --------------------------------------
  filtering(table, height, width, loops, type, blocksize);
  // -----------------------------------------------------

  // apothikeush se arxeio ths alagmenhs eikonas
  int fd_changed_image;
  // char *changed_image;
  //
  // changed_image = (char*)malloc( (strlen(image))*sizeof(char) );
  // strcpy(changed_image, image);

  // fd_changed_image = open(changed_image, O_CREAT|O_WRONLY, 0644);
  fd_changed_image = open("filtered.raw", O_CREAT|O_WRONLY, 0644);

  if( fd_changed_image == -1 )
  {
    perror("open");
    free(type);
    free(image);
    free(table);
    // free(changed_image);
    return -1;
  }

  i = 0;
  j = 0;
  while(i < bytes)
  {
    if( (j = write(fd_changed_image, table, bytes - i)) == -1 )
    {
      perror("write");
      free(table);
      free(type);
      free(image);
      // free(changed_image);
      return -1;
    }
    i += j;
  }
  close(fd_changed_image);
  // free(changed_image);

  gettimeofday(&et, NULL);
  end_time = et.tv_sec * 1000000 + et.tv_usec;

  free(table);

  printf("height: %d, width: %d, loops: %d, type: %s and execution time: %.3f sec\n", height, width, loops, type, (end_time-start_time)/1000000.0);
  return 0;
}
