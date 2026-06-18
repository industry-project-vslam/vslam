#include <stdio.h>

int main() {
    int i = 1;

    while (i <= 5)
    {
        printf("%i\n", i);
        ++i;
    }

    printf("\n");

    for (int j = 0; j < 5; j++) {
        printf("%i\n", j);
    }

    printf("\n");

    int k;

    for (k = 0; k < 6; k++) {
        if (k == 2) {
            continue;
        }
        if (k == 4) {
            break;
        }
        printf("%d\n", k);
    }

    // int i = 0;

    // while (i < 10) {
    //     if (i == 4) {
    //         break;
    //     }
    //     printf("%d\n", i);
    //     i++;
    // }

    // int i = 0;

    // while (i < 10) {
    //     if (i == 4) {
    //         i++;
    //         continue;
    //     }
    //     printf("%d\n", i);
    //     i++;
    // }

    return 0;
}