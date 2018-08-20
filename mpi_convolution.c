#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//rgb or grey--height--width loops
void print_err(int err){
  if(err==-1)
    fprintf(stderr, "Not enough arguments-(Value of errno: %d)\n\n", err);
  else if(err==-2)
    fprintf(stderr, "Loops must be more than 0-(Value of errno: %d)\n\n", err);
  else if(err==-3)
    fprintf(stderr, "The processes should be an integer(>1) to the 2nd power-(Value of errno: %d)\n\n", err);
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

int check_info(int argc,char** argv,int size){
  int err=0;
  if(argc!=6){ print_err(-1); err=-1;  return err;}

  int loops=atoi(argv[5]);
  if(loops<=0){ print_err(-2); err=-2;}

  if(sqrt(size)*sqrt(size)!=size||size==1){ print_err(-3); err=-3;}

  char type[5];
  strcpy(type,argv[2]);
  if((strcmp(type,"RGB")!=0)&&(strcmp(type,"GREY")!=0))
  { print_err(-4); err=-4;}

  int height=atoi(argv[3]),width=atoi(argv[4]);
  if((height%(int)sqrt(size)!=0)||(width%(int)sqrt(size)!=0)){ print_err(-5); err=-5;}

  FILE *image_raw;
  image_raw = fopen(argv[1], "rb");
  if (!image_raw){print_err(-6); err=-6;}
  else{
    fseek(image_raw, 0L, SEEK_END);
    if(!strcmp(type,"RGB")){
      if(ftell(image_raw)!=3*height*width){ print_err(-7); err=-7;}
    }
    else
      if(ftell(image_raw)!=height*width){ print_err(-7); err=-7;}

    fclose(image_raw);
  }

  return err;
}

int main (int argc, char* argv[])
{
  int rank, size;
  int height,width,loops;
  char type[5];
  int i;
  int image;
  int  err[1];
  MPI_Request req;
  MPI_Status sta;
  MPI_File image_File;

  MPI_Init (&argc, &argv);      /* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);        /* get number of processes */
  if(!rank){

    err[0]=check_info(argc,argv,size);

    for(i=1;i<size;i++)
      MPI_Send(err, 1, MPI_INT, i, 123, MPI_COMM_WORLD);
    if(err[0]!=0)
      MPI_Abort(MPI_COMM_WORLD, -1);
  }
  else{
     MPI_Irecv(err, 1, MPI_INT, 0, 123, MPI_COMM_WORLD, &req);
     MPI_Wait(&req, &sta);//wait for the process 0 to finish with checking arguments
     if(err[0]!=0)
        MPI_Finalize();
  }

  height=atoi(argv[3]);
  width=atoi(argv[4]);
  loops=atoi(argv[5]);

  MPI_File_open( MPI_COMM_WORLD, argv[1],MPI_MODE_RDONLY, MPI_INFO_NULL, &image_File );


  int block_heigth=height/sqrt(size);
  int block_width=width/sqrt(size);

  unsigned char **image_array;

  image_array=(unsigned char **)malloc((block_heigth+2)*sizeof(unsigned char *));

  if(!strcmp(argv[2],"RGB"))
    for(i=0;i<block_heigth+2;i++)
      image_array[i]=(unsigned char *)malloc(3*(block_width+2)*sizeof(unsigned char ));
  else
    for(i=0;i<block_heigth+2;i++)
      image_array[i]=(unsigned char *)malloc((block_width+2)*sizeof(unsigned char ));

      for(int k=0;k<block_heigth;k++){
        for(int l=0;l<block_width+2;l++){
          image_array[k][l]=0;
        }
      }

  if(!strcmp(argv[2],"RGB"))
    for(int k=0;k<block_heigth-1;k++){
      MPI_File_seek(image_File,3*(k*width+(rank/(int)sqrt(size))*block_heigth*width+(rank%(int)sqrt(size))*block_width), MPI_SEEK_SET );
      MPI_File_read(image_File,&image_array[k+1][3] ,3*block_width, MPI_BYTE, &sta );
    }
  else
    for(int k=0;k<block_heigth-1;k++){
      MPI_File_seek(image_File,k*width+(rank/(int)sqrt(size))*block_heigth*width+(rank%(int)sqrt(size))*block_width, MPI_SEEK_SET );
      MPI_File_read(image_File,&image_array[k+1][1], block_width, MPI_BYTE, &sta );
    }


  unsigned char * buffer_send;
  unsigned char * buffer_recv;

  int north,north_east,east,south_east,south,south_west,west,north_west;
  north=rank-sqrt(size);
  north_east=rank-sqrt(size)+1;
  east=rank+1;
  south_east=rank+sqrt(size)+1;
  south=rank+sqrt(size);
  south_west=rank+sqrt(size)-1;
  west=rank-1;
  north_west=rank-sqrt(size)-1;

  if(block_heigth>block_width){
    buffer_send=(unsigned char *)malloc((block_heigth)*sizeof(unsigned char ));
    buffer_recv=(unsigned char *)malloc((block_heigth)*sizeof(unsigned char ));
  }
  else{
    buffer_send=(unsigned char *)malloc((block_width)*sizeof(unsigned char ));
    buffer_recv=(unsigned char *)malloc((block_width)*sizeof(unsigned char ));
  }
  //send RGB
  if(!strcmp(argv[2],"RGB"))
  {
    if(north>=0&&north<size){
      buffer_send=image_array[1][3]
      MPI_Send(buffer_send, 10, MPI_BYTE, 1, 1, MPI_COMM_WORLD);
    }
    if(north_east>=0&&north_east<size){
      buffer_send=image_array[1][3*(block_width-1)]
      MPI_Send(buffer_send, 10, MPI_BYTE, 1, 2, MPI_COMM_WORLD);
    }
    if(east>=0&&east<size){
      for(int i=1;i<block_heigth;i++)
       buffer_send=image_array[i][3*(block_width-1)]
      MPI_Send(buffer_send, 10, MPI_BYTE, 1, 3, MPI_COMM_WORLD);
    }
    if(south_east>=0&&south_east<size){
      buffer_send=image_array[block_heigth-2][3*(block_width-1)]
      MPI_Send(buffer_send, 10, MPI_BYTE, 1, 4, MPI_COMM_WORLD);
    }
    if(south>=0&&south<size){
      MPI_Send(buffer_send, 10, MPI_BYTE, 1, 5, MPI_COMM_WORLD);
    }
    if(south_west>=0&&south_west<size){
      MPI_Send(buffer_send, 10, MPI_BYTE, 1, 6, MPI_COMM_WORLD);
    }
    if(west>=0&&west<size){
      MPI_Send(buffer_send, 10, MPI_BYTE, 1, 7, MPI_COMM_WORLD);
    }
    if(north_west>=0&&north_west<size){
      MPI_Send(buffer_send, 10, MPI_BYTE, 1, 8, MPI_COMM_WORLD);
    }
  }

 MPI_File_close( &image_File );


  for(i=0;i<block_heigth;i++)
    free(image_array[i]);
  free(image_array);
  MPI_Finalize(); /* finish MPI */
  return 0;
}

  //FILE *f = fopen("image4.raw", "wb");

/*
  for(int k=1;k<block_heigth+2-1;k++){
      for(int l=3;l<3*(block_width+2)-3;l++){
        putc(image_array[k][l], f);
      }
    }

    for(int k=1;k<block_heigth+2-1;k++){
        for(int l=1;l<(block_width+2)-1;l++){
          putc(image_array[k][l], f);
        }
      }*/
