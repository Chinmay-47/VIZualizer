import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class HomeService {

  constructor(private http: HttpClient) { }

  getSupervisedModels()
  {
    return this.http.get("http://192.168.1.8:3001/home/get-topics-list");
  }
}
