import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class LinearRegressionService {

  constructor(private http: HttpClient) { }

  getPlots(obj: any) {
    return this.http.post("http://192.168.1.8:3001/get-linear-regression-plots", obj);
  }

  getAnimation() {
    return this.http.get("http://192.168.1.8:3001/get-linear-regression-animation");
  }
}
