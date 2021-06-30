import { Component, OnInit } from '@angular/core';
import {NavService} from '../../../nav.service';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.scss']
})
export class HeaderComponent implements OnInit {

  constructor(public navService: NavService) { }

  ngOnInit(): void {
  }

}
