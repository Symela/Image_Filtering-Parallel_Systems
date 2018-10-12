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
  double start ,time_s;
  MPI_Request req;
  MPI_Status sta;
  MPI_File image_File,output_File;
  MPI_Datatype corner, line, column;

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

//uint8_t **image_array;
  uint8_t *image_array;
  uint8_t *image_conv;
  uint8_t *temp;

  int new_block_heigth;
  int new_block_width;

  //image_array=(uint8_t **)malloc((block_heigth+2)*sizeof(uint8_t *));

  //uint8_t image_array[block_heigth+2][block_width+2];
  //uint8_t image_array[block_heigth+2][3*(block_width+2)];

  if(!strcmp(argv[2],"RGB")){

    image_array=( uint8_t *)malloc((block_heigth+2)*3*(block_width+2)*sizeof(uint8_t ));
    image_conv=( uint8_t *)malloc((block_heigth+2)*3*(block_width+2)*sizeof(uint8_t ));

  //  for(i=0;i<block_heigth+2;i++)
  //    image_array[i]=(uint8_t *)malloc(3*(block_width+2)*sizeof(uint8_t ));

    for(int k=0;k<(block_heigth+2)*(3*(block_width+2));k++){
        image_array[k]=0;
    }

    new_block_heigth=block_heigth+2;
    new_block_width=3*(block_width+2);

  }
  else{

    image_array=( uint8_t *)malloc((block_heigth+2)*(block_width+2)*sizeof(uint8_t ));
    image_conv=( uint8_t *)malloc((block_heigth+2)*(block_width+2)*sizeof(uint8_t ));

  //  for(i=0;i<block_heigth+2;i++)
  //    image_array[i]=(uint8_t *)malloc((block_width+2)*sizeof(uint8_t ));

    for(int k=0;k<(block_heigth+2)*(block_width+2);k++){
        image_array[k]=0;
    }

    new_block_heigth=block_heigth+2;
    new_block_width=block_width+2;

  }

  if(!strcmp(argv[2],"RGB"))
    for(int k=0;k<new_block_heigth-2;k++){
      MPI_File_seek(image_File,3*(k*width+(rank/(int)sqrt(size))*block_heigth*width+(rank%(int)sqrt(size))*block_width), MPI_SEEK_SET );
    //  MPI_File_read(image_File,&image_array[k][3] ,3*block_width, MPI_BYTE, &sta );
      MPI_File_read(image_File,&image_array[(k+1)*new_block_width+3] ,3*block_width, MPI_BYTE, &sta );
    }
  else
    for(int k=0;k<new_block_heigth-2;k++){
      MPI_File_seek(image_File,k*width+(rank/(int)sqrt(size))*block_heigth*width+(rank%(int)sqrt(size))*block_width, MPI_SEEK_SET );
    //  MPI_File_read(image_File,&image_array[k][1], block_width, MPI_BYTE, &sta );
      MPI_File_read(image_File,&image_array[(k+1)*(block_width+2)+1] ,block_width, MPI_BYTE, &sta );

    }


  int north,north_east,east,south_east,south,south_west,west,north_west;

    if(rank%(int)sqrt(size)!=0)
      north_west=rank-sqrt(size)-1;
    else
      north_west=-1;

    if((rank)%(int)sqrt(size)!=0)
      west=rank-1;
    else
      west=-1;

    if((rank)%(int)sqrt(size)!=0)
      south_west=rank+sqrt(size)-1;
    else
      south_west=-1;

    if(rank%(int)sqrt(size)!=(int)sqrt(size)-1)
      north_east=rank-sqrt(size)+1;
    else
      north_east=-1;

    if(rank%(int)sqrt(size)!=(int)sqrt(size)-1)
      east=rank+1;
    else
      east=-1;

    if(rank%(int)sqrt(size)!=(int)sqrt(size)-1)
      south_east=rank+sqrt(size)+1;
    else
      south_east=-1;


    south=rank+sqrt(size);

    north=rank-sqrt(size);

  //taken from wikipedia
  float identity[3][3]= {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
  float edge_detection1[3][3] = {{1, 0, -1}, {0, 0, 0}, {-1, 0, 1}};
  float edge_detection2[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
  float edge_detection3[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
  float sharpen[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
  float box_blur[3][3] = {{1.0/9, 1.0/9, 1.0/9}, {1.0/9, 1.0/9, 1.0/9}, {1.0/9, 1.0/9, 1.0/9}};
  float gaussian_blur[3][3] = {{1.0/16, 2.0/16, 1.0/16}, {2.0/16, 4.0/16, 2.0/16}, {1.0/16, 2.0/16, 1.0/16}};
  float kernel[3][3];


  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      kernel[i][j]=gaussian_blur[2-i][2-j];

  MPI_Status status;

  if(!strcmp(argv[2],"RGB")){

    MPI_Type_vector(1, 3, 1, MPI_BYTE, &corner);
    MPI_Type_commit(&corner);
    MPI_Type_vector(new_block_width,1, 1, MPI_BYTE, &line);
    MPI_Type_commit(&line);
    MPI_Type_vector(new_block_heigth, 3,new_block_width, MPI_BYTE, &column);
    MPI_Type_commit(&column);


    uint8_t se[3];
    uint8_t sw[3];
    for(int i=0;i<3;i++){
      se[i]=-1;
      sw[i]=-1;
    }
     start = MPI_Wtime();
for(int i=0;i<loops;i++){
    //send to neighbours
    if(north>=0&&north<size){
      //    MPI_Send(&image_array[1][3], 1, line, north, 1, MPI_COMM_WORLD);
          MPI_Send(&image_array[1*(new_block_width)+3], 1, line, north, 1, MPI_COMM_WORLD);
    }
       if(north_east>=0&&north_east<size){
      //    MPI_Send(&image_array[1][new_block_width-6], 1, corner, north_east, 2, MPI_COMM_WORLD);
          MPI_Send(&image_array[1*(new_block_width)+(new_block_width-6)], 1, corner, north_east, 2, MPI_COMM_WORLD);
        }
        if(east>=0&&east<size){
    //      MPI_Send(&image_array[1][new_block_width-6], 1, column, east, 3, MPI_COMM_WORLD);
          MPI_Send(&image_array[1*(new_block_width)+(new_block_width-6)], 1, column, east, 3, MPI_COMM_WORLD);
        }
        if(south_east>=0&&south_east<size){
    //      MPI_Send(&image_array[new_block_heigth-2][new_block_width-6], 1, corner, south_east,4, MPI_COMM_WORLD);
          MPI_Send(&image_array[(new_block_heigth-2)*(new_block_width)+(new_block_width-6)], 1, corner, south_east,4, MPI_COMM_WORLD);
        }
        if(south>=0&&south<size){
    //      MPI_Send(&image_array[new_block_heigth-2][3], 1, line, south, 5, MPI_COMM_WORLD);
          MPI_Send(&image_array[(new_block_heigth-2)*(new_block_width)+3], 1, line, south, 5, MPI_COMM_WORLD);
        }
        if(south_west>=0&&south_west<size){
      //    MPI_Send(&image_array[new_block_heigth-2][3], 1, corner, south_west,6, MPI_COMM_WORLD);
          MPI_Send(&image_array[(new_block_heigth-2)*(new_block_width)+3], 1, corner, south_west,6, MPI_COMM_WORLD);
        }
        if(west>=0&&west<size){
      //    MPI_Send(&image_array[1][3], 1, column, west, 7, MPI_COMM_WORLD);
          MPI_Send(&image_array[1*(new_block_width)+3], 1, column, west, 7, MPI_COMM_WORLD);
        }
        if(north_west>=0&&north_west<size){
    //      MPI_Send(&image_array[1][3], 1, corner, north_west, 8, MPI_COMM_WORLD);
          MPI_Send(&image_array[1*(new_block_width)+3], 1, corner, north_west, 8, MPI_COMM_WORLD);
        }

        //recv from neighbours
        if(north>=0&&north<size){
      //    MPI_Recv(&image_array[0][3], 1, line, north, 5, MPI_COMM_WORLD, &status);
          MPI_Recv(&image_array[3], 1, line, north, 5, MPI_COMM_WORLD, &status);
        }
        if(north_east>=0&&north_east<size){
    //      MPI_Recv(&image_array[0][new_block_width-3], 1, corner, north_east, 6, MPI_COMM_WORLD, &status);
          MPI_Recv(&image_array[new_block_width-3], 1, corner, north_east, 6, MPI_COMM_WORLD, &status);
        }
        if(east>=0&&east<size){
    //      MPI_Recv(&image_array[1][new_block_width-3], 1, column, east, 7, MPI_COMM_WORLD, &status);
          MPI_Recv(&image_array[1*(new_block_width)+new_block_width-3], 1, column, east, 7, MPI_COMM_WORLD, &status);
        }
        if(south_east>=0&&south_east<size){
          MPI_Recv(&se, 1, corner, south_east, 8, MPI_COMM_WORLD, &status);
        }
        if(south>=0&&south<size){
      //    MPI_Recv(&image_array[new_block_heigth-1][3], 1, line, south, 1, MPI_COMM_WORLD, &status);
          MPI_Recv(&image_array[(new_block_heigth-1)*(new_block_width)+3], 1, line, south, 1, MPI_COMM_WORLD, &status);
        }
        if(south_west>=0&&south_west<size){
          MPI_Recv(&sw, 1, corner, south_west, 2, MPI_COMM_WORLD, &status);
        }
        if(west>=0&&west<size){
    //      MPI_Recv(&image_array[1][0], 1, column, west, 3, MPI_COMM_WORLD, &status);
          MPI_Recv(&image_array[new_block_width], 1, column, west, 3, MPI_COMM_WORLD, &status);
        }
        if(north_west>=0&&north_west<size){
    //      MPI_Recv(&image_array[0][0], 1, corner, north_west, 4, MPI_COMM_WORLD, &status);
          MPI_Recv(&image_array[0], 1, corner, north_west, 4, MPI_COMM_WORLD, &status);
        }



        if(se[0]!=-1||se[1]!=-1||se[2]!=-1){
          //  image_array[new_block_heigth-1][new_block_width-3]=se[0];
              image_array[(new_block_heigth-1)*(new_block_width)+new_block_width-3]=se[0];
        //    image_array[new_block_heigth-1][new_block_width-2]=se[1];
              image_array[(new_block_heigth-1)*(new_block_width)+new_block_width-2]=se[1];
        //    image_array[new_block_heigth-1][new_block_width-1]=se[2];
              image_array[(new_block_heigth-1)*(new_block_width)+new_block_width-1]=se[2];
        }
        if(sw[0]!=-1||sw[1]!=-1||sw[2]!=-1){
      //      image_array[new_block_heigth-1][0]=sw[0];
              image_array[(new_block_heigth-1)*(new_block_width)]=se[0];
      //      image_array[new_block_heigth-1][1]=sw[1];
              image_array[(new_block_heigth-1)*(new_block_width)+1]=se[1];
      //      image_array[new_block_heigth-1][2]=sw[2];
              image_array[(new_block_heigth-1)*(new_block_width)+2]=se[2];
        }

        for(int i=1;i<new_block_heigth-1;i++)
          for(int j=3;j<new_block_width-3;j++){
            image_conv[i*new_block_width+j]=0;
            for(int k=0;k<3;k++)
              for(int l=0;l<3;l++){
                int s=0;
                if(north_east<0||north_east>=size){
                  if(i==1&&j>=new_block_width-6)
                    if(k==0&&l==2){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(north_west<0||north_west>=size){
                  if(i==1&&(j>=3||j<6))
                    if(k==0&&l==0){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(south_east<0||south_east>=size){
                  if(i==new_block_heigth-2&&j>=new_block_width-6)
                    if(k==2&&l==2){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }


                }
                else if(south_west<0||south_west>=size){
                  if(i==new_block_heigth-2&&(j>=3||j<6))
                    if(k==2&&l==0){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(north<0||north>=size){
                  if(i==1)
                    if(k==0){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(east<0||east>=size){
                  if(j>=new_block_width-6)
                    if(l==2){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }
                }
                else if(south<0||south>=size){
                  if(i==new_block_heigth-2)
                    if(k==2){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(west<0||west>=size){
                  if(j<6)
                    if(l==0){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }
                }

                if(s==0)
                  image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j +(k-1)*new_block_width +(l-1)*3];
              }
          }

          temp=image_conv;
          image_conv=image_array;
          image_array=temp;
          int diff=0;
          for(int i=1;i<new_block_heigth-1;i++){
            for(int j=3;j<new_block_width-3;j++)
              if (image_conv[i*new_block_width+j]!=image_array[i*new_block_width+j]){
                diff++;
                break;
              }
            if(diff==1)
              break;
          }
          if(diff==0)
            break;

      }

  }
  else{

      MPI_Type_vector(1, 1, 1, MPI_BYTE, &corner);
      MPI_Type_commit(&corner);
      MPI_Type_vector(new_block_width, 1, 1, MPI_BYTE, &line);
      MPI_Type_commit(&line);
      MPI_Type_vector(new_block_heigth, 1, new_block_width, MPI_BYTE, &column);
      MPI_Type_commit(&column);


    uint8_t se=-1;
    uint8_t sw=-1;
    start = MPI_Wtime();

    for(int i=0;i<loops;i++){

    //send to neighbours
      if(north>=0&&north<size){
  //     MPI_Send(&image_array[1][1], 1, line, north, 1, MPI_COMM_WORLD);
       MPI_Send(&image_array[1*(block_width+2)+1], 1, line, north, 1, MPI_COMM_WORLD);
      }
      if(north_east>=0&&north_east<size){
  //      MPI_Send(&image_array[1][new_block_width-2], 1, corner, north_east, 2, MPI_COMM_WORLD);
        MPI_Send(&image_array[1*(block_width+2)+new_block_width-2], 1, corner, north_east, 2, MPI_COMM_WORLD);
      }
      if(east>=0&&east<size){
    //    MPI_Send(&image_array[1][new_block_width-2], 1, column, east, 3, MPI_COMM_WORLD);
        MPI_Send(&image_array[1*(block_width+2)+new_block_width-2], 1, column, east, 3, MPI_COMM_WORLD);
      }
      if(south_east>=0&&south_east<size){
    //    MPI_Send(&image_array[new_block_heigth-2][new_block_width-2], 1, corner, south_east,4, MPI_COMM_WORLD);
        MPI_Send(&image_array[(new_block_heigth-2)*(block_width+2)+new_block_width-2], 1, corner, south_east,4, MPI_COMM_WORLD);
      }
      if(south>=0&&south<size){
    //    MPI_Send(&image_array[new_block_heigth-2][1], 1, line, south, 5, MPI_COMM_WORLD);
        MPI_Send(&image_array[(new_block_heigth-2)*(block_width+2)+1], 1, line, south, 5, MPI_COMM_WORLD);
      }
      if(south_west>=0&&south_west<size){
    //    MPI_Send(&image_array[new_block_heigth-2][1], 1, corner, south_west,6, MPI_COMM_WORLD);
        MPI_Send(&image_array[(new_block_heigth-2)*(block_width+2)+1], 1, corner, south_west,6, MPI_COMM_WORLD);
      }
      if(west>=0&&west<size){
    //    MPI_Send(&image_array[1][1], 1, column, west, 7, MPI_COMM_WORLD);
        MPI_Send(&image_array[1*(block_width+2)+1], 1, column, west, 7, MPI_COMM_WORLD);
      }
      if(north_west>=0&&north_west<size){
  //      MPI_Send(&image_array[1][1], 1, corner, north_west, 8, MPI_COMM_WORLD);
        MPI_Send(&image_array[1*(block_width+2)+1], 1, corner, north_west, 8, MPI_COMM_WORLD);
      }

      //recv from neighbours
      if(north>=0&&north<size){
  //      MPI_Recv(&image_array[0][1], 1, line, north, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(&image_array[1], 1, line, north, 5, MPI_COMM_WORLD, &status);
      }
      if(north_east>=0&&north_east<size){
  //      MPI_Recv(&image_array[0][new_block_width-1], 1, corner, north_east, 6, MPI_COMM_WORLD, &status);
        MPI_Recv(&image_array[new_block_width-1], 1, corner, north_east, 6, MPI_COMM_WORLD, &status);
    }
      if(east>=0&&east<size){
  //      MPI_Recv(&image_array[1][new_block_width-1], 1, column, east, 7, MPI_COMM_WORLD, &status);
        MPI_Recv(&image_array[1*(block_width+2)+new_block_width-1], 1, column, east, 7, MPI_COMM_WORLD, &status);
      }
      if(south_east>=0&&south_east<size){
        MPI_Recv(&se, 1, corner, south_east, 8, MPI_COMM_WORLD, &status);
      }
      if(south>=0&&south<size){
  //      MPI_Recv(&image_array[new_block_heigth-1][1], 1, line, south, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&image_array[(new_block_heigth-1)*(block_width+2)+1], 1, line, south, 1, MPI_COMM_WORLD, &status);
      }
      if(south_west>=0&&south_west<size){
        MPI_Recv(&sw, 1, corner, south_west, 2, MPI_COMM_WORLD, &status);
      }
      if(west>=0&&west<size){
  //      MPI_Recv(&image_array[1][0], 1, column, west, 3, MPI_COMM_WORLD, &status);
        MPI_Recv(&image_array[1*(block_width+2)], 1, column, west, 3, MPI_COMM_WORLD, &status);
      }
      if(north_west>=0&&north_west<size){
  //      MPI_Recv(&image_array[0][0], 1, corner, north_west, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(&image_array[0], 1, corner, north_west, 4, MPI_COMM_WORLD, &status);
      }


        if(se!=-1){
      //    image_array[new_block_heigth-1][new_block_width-1]=se;
          image_array[(new_block_heigth-1)*(block_width+2)+new_block_width-1]=se;
        }
        if(sw!=-1){
      //    image_array[new_block_heigth-1][0]=sw;
          image_array[(new_block_heigth-1)*(block_width+2)]=sw;
        }

        for(int i=1;i<new_block_heigth-1;i++)
          for(int j=1;j<new_block_width-1;j++){
            image_conv[i*new_block_width+j]=0;
            for(int k=0;k<3;k++)
              for(int l=0;l<3;l++){
                int s=0;
                if(north_east<0||north_east>=size){
                  if(i==1&&j>=new_block_width-2)
                    if(k==0&&l==2){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(north_west<0||north_west>=size){
                  if(i==1&&(j>=1||j<2))
                    if(k==0&&l==0){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(south_east<0||south_east>=size){
                  if(i==new_block_heigth-2&&j>=new_block_width-2)
                    if(k==2&&l==2){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }


                }
                else if(south_west<0||south_west>=size){
                  if(i==new_block_heigth-2&&(j>=1||j<2))
                    if(k==2&&l==0){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(north<0||north>=size){
                  if(i==1)
                    if(k==0){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(east<0||east>=size){
                  if(j>=new_block_width-2)
                    if(l==2){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }
                }
                else if(south<0||south>=size){
                  if(i==new_block_heigth-2)
                    if(k==2){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }

                }
                else if(west<0||west>=size){
                  if(j<2)
                    if(l==0){
                      image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j];
                      s=1;
                    }
                }

                if(s==0)
                  image_conv[i*new_block_width+j]+=kernel[k][l]*image_array[i*new_block_width+j +(k-1)*new_block_width +(l-1)*1];
              }
          }

          temp=image_conv;
          image_conv=image_array;
          image_array=temp;
          int diff=0;
          for(int i=1;i<new_block_heigth-1;i++){
            for(int j=1;j<new_block_width-1;j++)
              if (image_conv[i*new_block_width+j]!=image_array[i*new_block_width+j]){
                diff++;
                break;
              }
            if(diff==1)
              break;
          }
          if(diff==0)
            break;

    }

  }


  int rc = MPI_File_open( MPI_COMM_WORLD, "out.raw", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &output_File );

  if(!strcmp(argv[2],"RGB"))
    MPI_File_preallocate(output_File,height*3*width);
  else
    MPI_File_preallocate(output_File,height*width);

  if(!strcmp(argv[2],"RGB"))
  for(int k=0;k<new_block_heigth-2;k++){
      MPI_File_write_at(output_File,3*(k*width+(rank/(int)sqrt(size))*block_heigth*width+(rank%(int)sqrt(size))*block_width),&image_array[(k+1)*new_block_width+3] ,3*block_width, MPI_BYTE, &sta );
    }
  else
  for(int k=0;k<new_block_heigth-2;k++){
      MPI_File_write_at(output_File,k*width+(rank/(int)sqrt(size))*block_heigth*width+(rank%(int)sqrt(size))*block_width,&image_array[(k+1)*(block_width+2)+1], block_width, MPI_BYTE, &sta );
    }
  MPI_File_close( &output_File );
  time_s =  MPI_Wtime() -  start  ;
  double max_time=0.0;
  double proc_time;

  if (rank)
    MPI_Send(&time_s, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  else{
    for (i = 1 ; i != size ; ++i) {
      MPI_Recv(&proc_time, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &sta);
      if (proc_time > max_time)
        max_time = proc_time;
    }
  }
  if (!rank)
  printf("Maximum time from all processes is: %f\n",max_time);

//  for(i=0;i<block_heigth;i++)
  //  free(image_array[i]);
  free(image_array);
  free(image_conv);

  MPI_Finalize(); /* finish MPI */
  return 0;
}
